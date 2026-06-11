#!/usr/bin/env python3
"""
Entry point for the DPO alignment stage.

Usage
-----
# Basic run — generates synthetic pairs from the SFT model
python dpo_train.py --config configs/dpo_config.yaml

# Use a pre-built preference dataset (skips generation)
python dpo_train.py --config configs/dpo_config.yaml \\
    --set data.preference_dataset_path=./outputs/deepseek-dsa-dpo/preference_data/synthetic_pairs \\
    --set data.generate_synthetic_preferences=false

# Tweak DPO algorithm hyperparams from the CLI
python dpo_train.py --config configs/dpo_config.yaml \\
    --set dpo.beta=0.2 \\
    --set dpo.loss_type=ipo \\
    --set training.learning_rate=3e-5

# Resume from a checkpoint
python dpo_train.py --config configs/dpo_config.yaml \\
    --set training.resume_from_checkpoint=./outputs/deepseek-dsa-dpo/checkpoint-200

Full two-stage pipeline
-----------------------
python train.py     --config configs/deepseek_dsa.yaml   # Stage 1 — SFT
python dpo_train.py --config configs/dpo_config.yaml     # Stage 2 — DPO
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.dpo_config import DPOConfig
from src.dpo_data_pipeline import build_preference_datasets
from src.dpo_model_setup import build_dpo_models
from src.dpo_trainer import run_dpo_training
from src.utils import get_logger, log_gpu_memory, set_seed, timer



# CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DPO alignment stage for DeepSeek Coder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dpo_config.yaml",
        help="Path to the DPO YAML configuration file.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="section.key=value",
        help="Override a config value, e.g. --set dpo.beta=0.2",
    )
    return parser.parse_args()


def parse_overrides(raw: list[str]) -> dict:
    out: dict = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Override must be in 'section.key=value' format, got: {item!r}")
        key, _, raw_val = item.partition("=")
        for caster in (int, float):
            try:
                raw_val = caster(raw_val)
                break
            except ValueError:
                pass
        if raw_val in ("true", "True"):
            raw_val = True
        elif raw_val in ("false", "False"):
            raw_val = False
        elif raw_val in ("null", "None", "none"):
            raw_val = None
        out[key] = raw_val
    return out


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    args = parse_args()

    cfg = DPOConfig.from_yaml(args.config)
    if args.overrides:
        cfg.apply_cli_overrides(parse_overrides(args.overrides))

    cfg.ensure_output_dirs()

    logger = get_logger("dpo", log_dir=f"{cfg.training.output_dir}/logs")
    logger.info("DPO config loaded from: %s", args.config)

    set_seed(cfg.training.seed)

    # 1. Load models
    log_gpu_memory(logger, prefix="Before DPO model load — ")
    with timer("DPO model setup", logger):
        policy_model, ref_model, tokenizer = build_dpo_models(cfg)

    # 2. Build / load preference dataset
    #    Pass policy_model only if synthetic generation is needed;
    #    it acts as the SFT model that generates candidate responses.
    generation_model = (
        policy_model
        if cfg.data.generate_synthetic_preferences
        and not cfg.data.preference_dataset_path
        and not cfg.data.preference_dataset_name
        else None
    )

    with timer("DPO data pipeline", logger):
        datasets = build_preference_datasets(
            cfg=cfg,
            tokenizer=tokenizer,
            model=generation_model,
        )

    logger.info(
        "Preference dataset — train: %d  val: %d",
        len(datasets["train"]),
        len(datasets["validation"]),
    )

    # 3. Train
    run_dpo_training(
        cfg=cfg,
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
    )


if __name__ == "__main__":
    main()
