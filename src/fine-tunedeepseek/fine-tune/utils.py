"""
Shared utilities: structured logging, reproducibility seeding,
GPU memory reporting, and a simple timing context manager.
"""

from __future__ import annotations

import gc
import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Generator

import numpy as np
import torch


# Logger

def get_logger(name: str, log_dir: str | None = None) -> logging.Logger:
    """
    Returns a logger that writes INFO+ to stdout and (optionally) to a
    rotating file under *log_dir*.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# Reproducibility

def set_seed(seed: int) -> None:
    """Pin all RNG sources for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes CuDNN deterministic at a slight speed cost
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


#
# GPU / memory helpers

def get_gpu_memory_summary() -> dict[str, str]:
    """Return per-device VRAM usage as human-readable strings."""
    if not torch.cuda.is_available():
        return {}
    summary: dict[str, str] = {}
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        summary[f"GPU:{i}"] = (
            f"allocated={alloc:.2f}GB  reserved={reserved:.2f}GB  total={total:.2f}GB"
        )
    return summary


def log_gpu_memory(logger: logging.Logger, prefix: str = "") -> None:
    for device, info in get_gpu_memory_summary().items():
        logger.info("%s%s — %s", prefix, device, info)


def free_memory() -> None:
    """Aggressively free unused CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Returns (trainable_params, all_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def log_trainable_parameters(model: torch.nn.Module, logger: logging.Logger) -> None:
    trainable, total = count_trainable_parameters(model)
    pct = 100 * trainable / total if total else 0.0
    logger.info(
        "Trainable params: %s / %s  (%.4f%%)",
        f"{trainable:,}",
        f"{total:,}",
        pct,
    )

# Timing

@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Generator[None, None, None]:
    """Context manager that logs elapsed wall-clock time for a block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        msg = f"[{label}] finished in {elapsed:.2f}s"
        if logger:
            logger.info(msg)
        else:
            print(msg)
