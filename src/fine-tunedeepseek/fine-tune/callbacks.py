"""
Custom Hugging Face Trainer callbacks.

Included callbacks
------------------
GpuMemoryCallback      — logs VRAM at each logging step
EarlyStoppingCallback  — stops training when eval metric stops improving
SampleGenerationCallback — generates a few completions at eval time so you
                           can visually inspect quality without leaving the run
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    GenerationConfig,
)

logger = logging.getLogger(__name__)


# GPU memory callback

class GpuMemoryCallback(TrainerCallback):
    """Logs peak GPU memory usage at every logging step."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            peak = torch.cuda.max_memory_allocated(i) / 1024**3
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            logger.info(
                "GPU:%d  allocated=%.2fGB  peak=%.2fGB",
                i,
                alloc,
                peak,
            )
            torch.cuda.reset_peak_memory_stats(i)


# Early stopping callback

class EarlyStoppingCallback(TrainerCallback):
    """
    Stops training when the tracked metric has not improved for
    *patience* evaluations.

    Parameters
    ----------
    patience        : int   — number of evals without improvement before stop
    min_delta       : float — minimum change counted as an improvement
    greater_is_better: bool — True for accuracy-like metrics, False for loss
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 1e-4,
        greater_is_better: bool = False,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.greater_is_better = greater_is_better
        self._best: Optional[float] = None
        self._wait = 0

    def _is_improvement(self, current: float) -> bool:
        if self._best is None:
            return True
        delta = current - self._best
        return delta > self.min_delta if self.greater_is_better else delta < -self.min_delta

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float],
        **kwargs,
    ) -> None:
        metric_key = args.metric_for_best_model
        if not metric_key.startswith("eval_"):
            metric_key = f"eval_{metric_key}"
        current = metrics.get(metric_key)
        if current is None:
            logger.warning("EarlyStoppingCallback: metric '%s' not found.", metric_key)
            return

        if self._is_improvement(current):
            self._best = current
            self._wait = 0
            logger.info(
                "EarlyStopping: new best %s = %.6f",
                metric_key,
                current,
            )
        else:
            self._wait += 1
            logger.info(
                "EarlyStopping: no improvement (%d / %d). best=%.6f  current=%.6f",
                self._wait,
                self.patience,
                self._best,
                current,
            )
            if self._wait >= self.patience:
                logger.warning(
                    "EarlyStopping triggered after %d evals without improvement.",
                    self.patience,
                )
                control.should_training_stop = True


# Sample generation callback

_DEFAULT_PROMPTS = [
    "Write a Python function to check if a string is a palindrome.",
    "Implement binary search iteratively in Python.",
    "Given an array, find the two numbers that sum to a target. Return their indices.",
]


class SampleGenerationCallback(TrainerCallback):
    """
    Generates a handful of completions at each eval step so you can
    eyeball quality without leaving the training run.

    Parameters
    ----------
    tokenizer    : the tokenizer
    prompts      : list of raw user-turn strings to complete
    every_n_evals: how often to generate (default: every eval)
    max_new_tokens: generation budget per sample
    """

    def __init__(
        self,
        tokenizer,
        prompts: Optional[list[str]] = None,
        every_n_evals: int = 1,
        max_new_tokens: int = 256,
    ):
        self.tokenizer = tokenizer
        self.prompts = prompts or _DEFAULT_PROMPTS
        self.every_n_evals = every_n_evals
        self.max_new_tokens = max_new_tokens
        self._eval_count = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs,
    ) -> None:
        self._eval_count += 1
        if self._eval_count % self.every_n_evals != 0:
            return

        model.eval()
        gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        logger.info("=" * 60)
        logger.info("Sample generations at step %d:", state.global_step)

        for i, prompt in enumerate(self.prompts, 1):
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    generation_config=gen_cfg,
                )

            # Only decode the newly generated tokens
            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(new_ids, skip_special_tokens=True)

            logger.info("\n[Sample %d]\nPrompt : %s\nResponse: %s\n", i, prompt, response)

        logger.info("=" * 60)
        model.train()
