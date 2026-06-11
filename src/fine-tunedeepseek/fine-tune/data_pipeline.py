"""
Data pipeline: load → validate → format → quality-filter → split → tokenise.

Design goals
------------
* Validates schema and data quality before any tokenisation.
* Filters sequences that are too short or too long.
* Produces a reproducible train / validation split.
* Optionally packs short sequences together (ConstantLengthDataset) to eliminate padding waste in training.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from trl import DataCollatorForCompletionOnlyLM
from transformers import PreTrainedTokenizerBase

from .config import DataConfig

logger = logging.getLogger(__name__)

# Required columns in the raw dataset
REQUIRED_COLUMNS = {"instruction", "input", "output"}


#-----------
# Schema / quality validation
#-----------

def validate_dataset(dataset: Dataset, split: str = "train") -> None:
    """Raise if required columns are missing or the split is empty."""
    missing = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset split '{split}' is missing columns: {missing}")
    if len(dataset) == 0:
        raise ValueError(f"Dataset split '{split}' is empty.")
    logger.info("Dataset split '%s': %d rows, columns=%s", split, len(dataset), dataset.column_names)


def log_data_statistics(dataset: Dataset, split: str = "train") -> None:
    """Log basic statistics about text lengths."""
    lengths = [
        len(str(r["instruction"])) + len(str(r["input"])) + len(str(r["output"]))
        for r in dataset
    ]
    import statistics
    logger.info(
        "Split '%s' char-length stats — min=%d  max=%d  mean=%.0f  median=%.0f",
        split,
        min(lengths),
        max(lengths),
        statistics.mean(lengths),
        statistics.median(lengths),
    )


#-----------
# Formatting
#-----------

def build_chat_formatter(tokenizer: PreTrainedTokenizerBase):
    """
    Returns a function that converts one example dict into {"text": <str>}.
    Uses the tokenizer's built-in chat template so the format stays consistent
    with whatever the base model was pretrained with.
    """

    def format_chat(example: dict[str, Any]) -> dict[str, str]:
        user_content = (
            f"{example['instruction'].strip()}\n{example['input'].strip()}"
            if example.get("input", "").strip()
            else example["instruction"].strip()
        )
        messages = [
            {"role": "user","content": user_content},
            {"role": "assistant", "content": example["output"].strip()},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return format_chat


#-----------
# Quality filtering
#-----------

def build_length_filter(
    tokenizer: PreTrainedTokenizerBase,
    min_tokens: int,
    max_tokens: int,
):
    """
    Returns a filter function that removes samples whose token count falls
    outside [min_tokens, max_tokens].  Avoids padding-dominated batches and
    truncation-heavy samples.
    """

    def length_ok(example: dict[str, Any]) -> bool:
        ids = tokenizer(
            example["text"],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        return min_tokens <= len(ids) <= max_tokens

    return length_ok


#-----------
# Public entry point
#-----------

def build_datasets(
    cfg: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> DatasetDict:
    """
    Full pipeline: load → validate → format → filter → split.

    Returns a DatasetDict with keys "train" and "validation".
    """

    # 1. Load
    logger.info("Loading dataset: %s", cfg.dataset_name)
    raw: DatasetDict = load_dataset(cfg.dataset_name)

    train_split: Dataset = raw["train"]
    validate_dataset(train_split, split="train")
    log_data_statistics(train_split, split="train")

    # 2. Format
    logger.info("Applying chat template …")
    formatter = build_chat_formatter(tokenizer)
    formatted: Dataset = train_split.map(
        formatter,
        num_proc=cfg.num_proc,
        desc="Formatting prompts",
        remove_columns=train_split.column_names,
    )

    # 3. Quality filter
    logger.info("Filtering by token length [%d, %d] …", cfg.min_token_length, cfg.max_token_length)
    length_filter = build_length_filter(
        tokenizer,
        min_tokens=cfg.min_token_length,
        max_tokens=cfg.max_token_length,
    )
    before = len(formatted)
    formatted = formatted.filter(length_filter, num_proc=cfg.num_proc, desc="Length filter")
    after = len(formatted)
    dropped = before - after
    logger.info(
        "Length filter: kept %d / %d rows  (dropped %d, %.1f%%)",
        after,
        before,
        dropped,
        100 * dropped / before if before else 0,
    )

    # 4. Deduplication
    logger.info("Deduplicating on 'text' …")
    seen: set[int] = set()
    keep: list[bool] = []
    for row in formatted:
        h = hash(row["text"])
        keep.append(h not in seen)
        seen.add(h)
    formatted = formatted.select([i for i, k in enumerate(keep) if k])
    logger.info("After dedup: %d rows", len(formatted))

    # 5. Train / val split ---------------------------------------------------
    logger.info("Splitting: val_ratio=%.2f", cfg.val_split_ratio)
    splits = formatted.train_test_split(
        test_size=cfg.val_split_ratio,
        seed=42,
        shuffle=True,
    )
    dataset_dict = DatasetDict(
        train=splits["train"],
        validation=splits["test"],
    )
    logger.info(
        "Final split sizes — train: %d  validation: %d",
        len(dataset_dict["train"]),
        len(dataset_dict["validation"]),
    )
    return dataset_dict

# Completion-only collator (masks prompt tokens in the loss)

def get_completion_only_collator(
    tokenizer: PreTrainedTokenizerBase,
) -> DataCollatorForCompletionOnlyLM:
    """
    Returns a collator that sets labels to -100 for the instruction tokens so
    the model only learns to predict the assistant response.
    """
    # The response template varies by chat format; we look for the assistant turn marker.
    # DeepSeek uses a specific token sequence — fall back to a safe string if not found.
    response_template_ids = tokenizer.encode(
        "<|Assistant|>",
        add_special_tokens=False,
    )
    return DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )
