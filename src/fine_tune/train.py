"""
Entry point for the DeepSeek DSA SFT pipeline.

Usage
-----
# Basic run with default YAML
python train.py --config configs/deepseek_dsa.yaml

# Override any config field from the command line
python train.py --config configs/deepseek_dsa.yaml \
    --set training.learning_rate=1e-4 \
    --set lora.r=32 \
    --set training.num_train_epochs=5

# Resume from a checkpoint
python train.py --config configs/deepseek_dsa.yaml \
    --set training.resume_from_checkpoint=./outputs/deepseek-dsa-sft/checkpoint-400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the package is importable when run from the project root
sys.path.insert(0, str(Path(__file__).parent))

from config import SFTConfig
from data_pipeline import build_datasets
from model_setup import build_model_and_tokenizer
from trainer import run_training
from utils import get_logger, set_seed, log_gpu_memory, timer


# CLI argument parsing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production SFT pipeline for DeepSeek Coder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "deepseek_dsa.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="section.key=value",
        help="Override a config value, e.g. --set training.learning_rate=1e-4",
    )
    return parser.parse_args()


def parse_overrides(raw: list[str]) -> dict:
    """Convert ['a.b=1', 'c.d=hello'] into {'a.b': 1, 'c.d': 'hello'}."""
    out: dict = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Override must be in 'section.key=value' format, got: {item!r}")
        key, _, raw_val = item.partition("=")
        # Attempt numeric coercion; keep as string otherwise
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


# Main


def main() -> None:
    args = parse_args()

    # 1. Load and patch config
    cfg = SFTConfig.from_yaml(args.config)
    if args.overrides:
        overrides = parse_overrides(args.overrides)
        cfg.apply_cli_overrides(overrides)

    cfg.ensure_output_dirs()

    logger = get_logger(
        "sft",
        log_dir=f"{cfg.training.output_dir}/logs",
    )
    logger.info("Config loaded from: %s", args.config)
    if args.overrides:
        logger.info("CLI overrides applied: %s", args.overrides)

    # 2. Reproducibility
    set_seed(cfg.training.seed)
    logger.info("Random seed set to %d", cfg.training.seed)

    # 3. Model & tokenizer
    log_gpu_memory(logger, prefix="Before model load — ")
    with timer("model setup", logger):
        model, tokenizer = build_model_and_tokenizer(cfg.model, cfg.lora)

    # 4. Dataset
    with timer("data pipeline", logger):
        datasets = build_datasets(cfg.data, tokenizer)

    logger.info(
        "Dataset sizes — train: %d  val: %d",
        len(datasets["train"]),
        len(datasets["validation"]),
    )

    # 5. Train
    run_training(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
    )


if __name__ == "__main__":
    main()
