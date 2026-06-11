"""
DPO Data Pipeline
Loads an existing preference dataset OR triggers synthetic generation,
validates schema, applies quality filters, and splits into train / val.

DPOTrainer expects rows with exactly these three string fields:
  prompt    — the user turn (plain text, NOT templated)
  chosen    — full assistant response that is preferred
  rejected  — full assistant response that is dispreferred
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

from .dpo_config import DPOConfig

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"prompt", "chosen", "rejected"}

# Validation helpers

def validate_preference_dataset(dataset: Dataset, split: str = "train") -> None:
    missing = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Preference dataset split '{split}' is missing columns: {missing}. "
            f"Got: {dataset.column_names}"
        )
    if len(dataset) == 0:
        raise ValueError(f"Preference dataset split '{split}' is empty.")
    logger.info(
        "Preference dataset '%s': %d rows, columns=%s",
        split, len(dataset), dataset.column_names,
    )


def log_preference_statistics(dataset: Dataset) -> None:
    chosen_lens   = [len(r["chosen"])   for r in dataset]
    rejected_lens = [len(r["rejected"]) for r in dataset]
    import statistics
    logger.info(
        "Chosen   length — min=%d  max=%d  mean=%.0f",
        min(chosen_lens), max(chosen_lens), statistics.mean(chosen_lens),
    )
    logger.info(
        "Rejected length — min=%d  max=%d  mean=%.0f",
        min(rejected_lens), max(rejected_lens), statistics.mean(rejected_lens),
    )

# Quality filter

def filter_trivial_pairs(dataset: Dataset, min_char_diff: int = 20) -> Dataset:
    """
    Remove pairs where chosen and rejected are nearly identical.
    Trivial pairs don't provide a useful training signal.
    """
    before = len(dataset)

    def is_nontrivial(example):
        return abs(len(example["chosen"]) - len(example["rejected"])) >= min_char_diff or \
               example["chosen"].strip() != example["rejected"].strip()

    dataset = dataset.filter(is_nontrivial, desc="Filtering trivial pairs")
    after = len(dataset)
    logger.info(
        "Trivial-pair filter: kept %d / %d  (removed %d)",
        after, before, before - after,
    )
    return dataset


def filter_by_token_length(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_prompt_length: int,
    max_seq_length: int,
) -> Dataset:
    """Keep only rows where both chosen and rejected fit within the context window."""

    def within_limits(example):
        prompt_ids = tokenizer(
            example["prompt"],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        chosen_ids = tokenizer(
            example["chosen"],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        rejected_ids = tokenizer(
            example["rejected"],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        return (
            len(prompt_ids) <= max_prompt_length
            and len(prompt_ids) + len(chosen_ids)   <= max_seq_length
            and len(prompt_ids) + len(rejected_ids) <= max_seq_length
        )

    before = len(dataset)
    dataset = dataset.filter(within_limits, desc="Token length filter")
    after = len(dataset)
    logger.info(
        "Token length filter: kept %d / %d rows",
        after, before,
    )
    return dataset


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_preference_datasets(
    cfg: DPOConfig,
    tokenizer: PreTrainedTokenizerBase,
    model=None,                      # required only for synthetic generation
) -> DatasetDict:
    """
    Full preference-data pipeline.
    Returns DatasetDict with keys "train" and "validation".
    """

    # 1. Source the raw preference data -------------------------------------
    dataset: Optional[Dataset] = None

    if cfg.data.preference_dataset_path:
        logger.info("Loading preference dataset from disk: %s", cfg.data.preference_dataset_path)
        dataset = load_from_disk(cfg.data.preference_dataset_path)
        if isinstance(dataset, DatasetDict):
            dataset = dataset["train"]

    elif cfg.data.preference_dataset_name:
        logger.info("Loading preference dataset from Hub: %s", cfg.data.preference_dataset_name)
        raw = load_dataset(cfg.data.preference_dataset_name)
        dataset = raw["train"]

    elif cfg.data.generate_synthetic_preferences:
        if model is None:
            raise ValueError(
                "A model must be provided to generate synthetic preference pairs."
            )
        from .preference_generator import generate_preference_dataset
        dataset = generate_preference_dataset(cfg, model, tokenizer)

    if dataset is None:
        raise RuntimeError("Could not obtain a preference dataset. Check your DPODataConfig.")

    # 2. Validate schema -----------------------------------------------------
    validate_preference_dataset(dataset)
    log_preference_statistics(dataset)

    # 3. Quality filters -----------------------------------------------------
    dataset = filter_trivial_pairs(dataset)
    dataset = filter_by_token_length(
        dataset,
        tokenizer,
        max_prompt_length=cfg.data.max_prompt_length,
        max_seq_length=cfg.data.max_seq_length,
    )

    if len(dataset) < 10:
        raise RuntimeError(
            f"Only {len(dataset)} preference pairs remain after filtering. "
            "This is too few to train on. Check your dataset or lower filter thresholds."
        )

    # 4. Train / val split ---------------------------------------------------
    splits = dataset.train_test_split(
        test_size=cfg.data.val_split_ratio,
        seed=cfg.training.seed,
        shuffle=True,
    )
    result = DatasetDict(train=splits["train"], validation=splits["test"])
    logger.info(
        "Final preference split — train: %d  validation: %d",
        len(result["train"]),
        len(result["validation"]),
    )
    return result
