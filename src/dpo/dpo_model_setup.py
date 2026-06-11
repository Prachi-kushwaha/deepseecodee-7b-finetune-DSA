"""
DPO model setup.

Key differences from SFT model setup
--------------------------------------
* Loads from the already fine-tuned SFT checkpoint (not the raw base model).
* Builds a frozen reference model for the DPO KL-penalty term.
* Two strategies for the reference model:
    - use_peft_ref_model=True  — re-use the same base model with adapters
      disabled at reference-forward-pass time (saves ~7 GB VRAM).
    - use_peft_ref_model=False — load a separate copy in bfloat16
      (slower, more VRAM, but simpler).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .dpo_config import DPOConfig, DPOLoRAConfig, DPOModelConfig
from .utils import free_memory, log_gpu_memory, log_trainable_parameters

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
    "float32":  torch.float32,
}


def _dtype(name: str) -> torch.dtype:
    return _DTYPE_MAP[name]


# Tokenizer

def load_tokenizer_for_dpo(cfg: DPOModelConfig) -> PreTrainedTokenizerBase:
    """Load tokenizer from the SFT checkpoint so special tokens are preserved."""
    logger.info("Loading tokenizer from SFT checkpoint: %s", cfg.sft_model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.sft_model_path,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"   # DPO trainer needs left-padding
    logger.info("Tokenizer loaded — vocab_size=%d", tokenizer.vocab_size)
    return tokenizer


# Policy model (trained during DPO)

def _build_bnb_config(cfg: DPOModelConfig) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_dtype(cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


def _load_kwargs(cfg: DPOModelConfig) -> dict:
    kwargs = dict(
        quantization_config=_build_bnb_config(cfg),
        device_map="auto",
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=_dtype(cfg.bnb_4bit_compute_dtype),
    )
    if cfg.use_flash_attention_2:
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 enabled for policy model")
        except ImportError:
            logger.warning("flash_attn not installed — using standard attention")
    return kwargs


def load_policy_model(cfg: DPOModelConfig, lora_cfg: DPOLoRAConfig) -> PreTrainedModel:
    """
    Load the SFT checkpoint as the DPO policy model and re-wrap it with
    a fresh set of LoRA adapters so the DPO gradients flow only through them.
    """
    logger.info("Loading policy model from: %s", cfg.sft_model_path)
    base = AutoModelForCausalLM.from_pretrained(cfg.sft_model_path, **_load_kwargs(cfg))

    if cfg.gradient_checkpointing:
        base.config.use_cache = False
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    peft_cfg = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )
    policy = get_peft_model(base, peft_cfg)
    log_trainable_parameters(policy, logger)
    return policy


# Reference model (frozen)

def load_reference_model(
    cfg: DPOModelConfig,
) -> Optional[PreTrainedModel]:
    """
    Build the frozen reference model.

    If use_peft_ref_model=True we return None — TRL's DPOTrainer will
    automatically use the policy model's base (adapters disabled) as the
    reference, halving VRAM usage.

    If use_peft_ref_model=False we load the SFT checkpoint separately in
    bfloat16 and freeze it.
    """
    if cfg.use_peft_ref_model:
        logger.info(
            "Reference model: using policy base with adapters disabled (VRAM-efficient)."
        )
        return None

    logger.info("Loading separate reference model from: %s", cfg.sft_model_path)
    ref = AutoModelForCausalLM.from_pretrained(
        cfg.sft_model_path,
        device_map="auto",
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=torch.bfloat16,
    )
    for param in ref.parameters():
        param.requires_grad = False
    ref.eval()
    logger.info("Reference model loaded and frozen.")
    return ref


# Public factory

def build_dpo_models(
    cfg: DPOConfig,
) -> Tuple[PreTrainedModel, Optional[PreTrainedModel], PreTrainedTokenizerBase]:
    """
    Returns (policy_model, reference_model, tokenizer).
    reference_model is None when use_peft_ref_model=True.
    """
    free_memory()

    tokenizer = load_tokenizer_for_dpo(cfg.model)
    log_gpu_memory(logger, prefix="Before DPO model load — ")

    policy = load_policy_model(cfg.model, cfg.lora)
    log_gpu_memory(logger, prefix="After policy model load — ")

    ref = load_reference_model(cfg.model)
    log_gpu_memory(logger, prefix="After reference model load — ")

    return policy, ref, tokenizer
