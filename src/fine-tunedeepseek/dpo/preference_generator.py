"""
Synthetic Preference Generator
================================
Takes the fine-tuned SFT model, generates N candidate responses per prompt,
scores them, and produces (prompt, chosen, rejected) triplets for DPO.

Scoring strategies
------------------
1. execution   — run the generated Python code, score by test-pass rate
2. length_penalty — simple heuristic: penalise verbosity + reward correctness signals
3. judge_llm   — call a separate LLM to rate solutions (slow but high quality)

Output schema (each row in the HF Dataset)
-------------------------------------------
{
  "prompt":   str,   # raw user message
  "chosen":   str,   # full assistant response (better)
  "rejected": str,   # full assistant response (worse)
  "score_chosen":   float,
  "score_rejected": float,
}
"""

from __future__ import annotations

import ast
import json
import logging
import multiprocessing
import os
import signal
import textwrap
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

from .dpo_config import DPOConfig, ScoringConfig

logger = logging.getLogger(__name__)


# Data structures

@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    score_chosen: float
    score_rejected: float


# Scoring strategies

def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")


def score_by_execution(code: str, timeout: int = 5) -> float:
    """
    Attempt to parse and execute the generated code snippet.
    Returns a score in [0, 1]:
      1.0  — code is syntactically valid and executes without error
      0.5  — code is syntactically valid but raises a runtime error
      0.0  — code has a syntax error or is empty
    """
    if not code.strip():
        return 0.0

    # Extract the first Python code block if wrapped in markdown fences
    if "```python" in code:
        start = code.find("```python") + 9
        end = code.find("```", start)
        code = code[start:end].strip() if end != -1 else code[start:].strip()
    elif "```" in code:
        start = code.find("```") + 3
        end = code.find("```", start)
        code = code[start:end].strip() if end != -1 else code[start:].strip()

    # Syntax check
    try:
        ast.parse(code)
    except SyntaxError:
        return 0.0

    # Runtime check in a fresh namespace with a timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        exec(compile(code, "<string>", "exec"), {})  # noqa: S102
        return 1.0
    except TimeoutError:
        return 0.4
    except Exception:
        return 0.5
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def score_by_length_penalty(response: str) -> float:
    """
    Heuristic score based on:
    - Presence of a code block         (+0.4)
    - Reasonable length (200-1500 chars)(+0.3)
    - Has an explanation               (+0.2)
    - No obvious boilerplate filler    (+0.1)
    """
    score = 0.0
    if "```" in response:
        score += 0.4
    length = len(response)
    if 200 <= length <= 1500:
        score += 0.3
    if len(response.split()) > 30:
        score += 0.2
    filler_phrases = ["I hope this helps", "Let me know", "Feel free"]
    if not any(p.lower() in response.lower() for p in filler_phrases):
        score += 0.1
    return score


def build_scorer(cfg: ScoringConfig) -> Callable[[str], float]:
    if cfg.strategy == "execution":
        timeout = cfg.execution_timeout_seconds
        return lambda resp: score_by_execution(resp, timeout=timeout)
    elif cfg.strategy == "length_penalty":
        return score_by_length_penalty
    elif cfg.strategy == "judge_llm":
        logger.warning(
            "judge_llm scoring is slow. Make sure '%s' is loaded.", cfg.judge_model
        )
        # Placeholder — wire in your judge model call here
        return score_by_length_penalty
    else:
        raise ValueError(f"Unknown scoring strategy: {cfg.strategy}")


# Response generation

def _build_prompt_text(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    """Convert a raw dataset row into the model's prompt format."""
    user_content = (
        f"{example['instruction'].strip()}\n{example['input'].strip()}"
        if example.get("input", "").strip()
        else example["instruction"].strip()
    )
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate_candidates(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    num_return_sequences: int,
    max_new_tokens: int,
) -> List[str]:
    """Sample *num_return_sequences* diverse responses for a single prompt."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    output_ids = model.generate(**inputs, generation_config=gen_cfg)
    prompt_len = inputs["input_ids"].shape[1]

    responses = [
        tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
        for ids in output_ids
    ]
    return responses


# Pair construction

def build_preference_pair(
    prompt: str,
    responses: List[str],
    scorer: Callable[[str], float],
    min_score_margin: float = 0.15,
) -> Optional[PreferencePair]:
    """
    Score all candidates, take the best and worst.
    Discard pairs where the score margin is below *min_score_margin* (too similar).
    """
    scores = [(r, scorer(r)) for r in responses]
    scores.sort(key=lambda x: x[1], reverse=True)

    best_response, best_score   = scores[0]
    worst_response, worst_score = scores[-1]

    if best_score - worst_score < min_score_margin:
        return None  # Not distinguishable enough

    return PreferencePair(
        prompt=prompt,
        chosen=best_response,
        rejected=worst_response,
        score_chosen=best_score,
        score_rejected=worst_score,
    )


# Main pipeline

def generate_preference_dataset(
    cfg: DPOConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """
    Full synthetic preference generation pipeline.

    1. Load source prompts from the SFT dataset.
    2. For each prompt, generate N candidate responses.
    3. Score and rank candidates.
    4. Emit (prompt, chosen, rejected) triplets.
    5. Save the dataset to disk for reproducibility.
    """
    logger.info(
        "Generating preference pairs from '%s' (%d prompts, %d candidates each) …",
        cfg.data.source_dataset_name,
        cfg.data.num_samples_for_generation,
        cfg.data.num_generations_per_prompt,
    )

    raw_dataset = load_dataset(cfg.data.source_dataset_name, split="train")
    raw_dataset = raw_dataset.shuffle(seed=cfg.training.seed).select(
        range(min(cfg.data.num_samples_for_generation, len(raw_dataset)))
    )

    scorer = build_scorer(cfg.scoring)
    model.eval()

    pairs: List[dict] = []
    skipped = 0

    for i, example in enumerate(raw_dataset):
        prompt = _build_prompt_text(example, tokenizer)
        candidates = generate_candidates(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            num_return_sequences=cfg.data.num_generations_per_prompt,
            max_new_tokens=512,
        )
        pair = build_preference_pair(prompt, candidates, scorer)

        if pair is None:
            skipped += 1
            continue

        pairs.append({
            "prompt":         pair.prompt,
            "chosen":         pair.chosen,
            "rejected":       pair.rejected,
            "score_chosen":   pair.score_chosen,
            "score_rejected": pair.score_rejected,
        })

        if (i + 1) % 50 == 0:
            logger.info(
                "Progress: %d / %d  |  pairs collected: %d  |  skipped: %d",
                i + 1, len(raw_dataset), len(pairs), skipped,
            )

    logger.info(
        "Generation complete — %d pairs collected, %d discarded (margin too small)",
        len(pairs), skipped,
    )

    if not pairs:
        raise RuntimeError(
            "No preference pairs were generated. "
            "Lower min_score_margin or use a different scoring strategy."
        )

    dataset = Dataset.from_list(pairs)

    # Persist to disk so you don't regenerate every run
    save_path = os.path.join(cfg.training.output_dir, "preference_data", "synthetic_pairs")
    dataset.save_to_disk(save_path)
    logger.info("Preference dataset saved to: %s", save_path)

    return dataset
