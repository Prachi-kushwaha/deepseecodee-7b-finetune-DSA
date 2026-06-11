"""
Model & tokenizer initialisation.

Handles:
  * BitsAndBytes 4-bit quantisation
  * Flash Attention 2 (graceful fallback)
  * LoRA / QLoRA adapter injection via PEFT
  * Gradient checkpointing
  * Proper pad-token setup
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from config import LoRAConfig, ModelConfig
from utils import free_memory, log_gpu_memory, log_trainable_parameters

logger = logging.getLogger(__name__)


# dtype resolution

_DTYPE_MAP = {
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
    "float32":  torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPE_MAP[name]
    except KeyError:
        raise ValueError(f"Unknown dtype '{name}'. Choose from {list(_DTYPE_MAP)}")


# Tokenizer

def load_tokenizer(cfg: ModelConfig) -> PreTrainedTokenizerBase:
    """
    Load and configure the tokenizer.
    * Sets pad_token to eos_token (standard for decoder-only models).
    * Ensures padding side is 'right' to avoid attention-mask issues.
    """
    logger.info("Loading tokenizer: %s", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )

    # Required for decoder-only models that lack a dedicated pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("pad_token set to eos_token: %r", tokenizer.pad_token)

    tokenizer.padding_side = "right"
    logger.info(
        "Tokenizer ready — vocab_size=%d  pad='%s'  eos='%s'",
        tokenizer.vocab_size,
        tokenizer.pad_token,
        tokenizer.eos_token,
    )
    return tokenizer


# BitsAndBytes config

def build_bnb_config(cfg: ModelConfig) -> BitsAndBytesConfig:
    compute_dtype = _resolve_dtype(cfg.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


# Base model

def load_base_model(
    cfg: ModelConfig,
    bnb_config: BitsAndBytesConfig,
) -> PreTrainedModel:
    """Load the quantised base model, optionally with Flash Attention 2."""
    logger.info("Loading base model: %s", cfg.model_name)

    kwargs: dict = dict(
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=_resolve_dtype(cfg.bnb_4bit_compute_dtype),
    )

    if cfg.use_flash_attention_2:
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled")
        except ImportError:
            logger.warning(
                "flash_attn not installed — falling back to standard attention. "
                "Install with: pip install flash-attn --no-build-isolation"
            )

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)

    if cfg.gradient_checkpointing:
        model.config.use_cache = False      # incompatible with grad checkpointing
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled")

    return model


# LoRA

def build_lora_config(cfg: LoRAConfig) -> LoraConfig:
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias=cfg.bias,
        task_type=cfg.task_type,
    )


def apply_lora(
    model: PreTrainedModel,
    lora_cfg: LoRAConfig,
    logger: logging.Logger,
) -> PreTrainedModel:
    """Prepare model for k-bit training then inject LoRA adapters."""
    logger.info("Preparing model for k-bit training …")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    lora_config = build_lora_config(lora_cfg)
    model = get_peft_model(model, lora_config)
    log_trainable_parameters(model, logger)
    return model


# Public factory

def build_model_and_tokenizer(
    model_cfg: ModelConfig,
    lora_cfg: LoRAConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    End-to-end model + tokenizer setup.
    Returns (peft_model, tokenizer).
    """
    free_memory()

    tokenizer = load_tokenizer(model_cfg)

    bnb_config = build_bnb_config(model_cfg)
    base_model = load_base_model(model_cfg, bnb_config)
    log_gpu_memory(logger, prefix="After base model load — ")

    peft_model = apply_lora(base_model, lora_cfg, logger)
    log_gpu_memory(logger, prefix="After LoRA injection — ")

    return peft_model, tokenizer
