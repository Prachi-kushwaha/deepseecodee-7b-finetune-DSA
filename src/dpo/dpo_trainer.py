"""
DPO Training Orchestrator
==========================
Wires together the DPOTrainer, custom callbacks, W&B logging,
reward-margin tracking, and post-training adapter merging.

Key DPO metrics logged automatically by TRL's DPOTrainer
  rewards/chosen       — average implicit reward for chosen responses
  rewards/rejected     — average implicit reward for rejected responses
  rewards/margins      — chosen − rejected  (the key alignment signal)
  rewards/accuracies   — fraction of pairs where chosen > rejected
  logps/chosen         — log-probs under the policy for chosen
  logps/rejected       — log-probs under the policy for rejected
  logits/chosen        — raw logits for chosen
  logits/rejected      — raw logits for rejected
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from peft import PeftModel
from transformers import TrainingArguments
from trl import DPOConfig as TRLDPOConfig
from trl import DPOTrainer

from .callbacks import EarlyStoppingCallback, GpuMemoryCallback
from .dpo_config import DPOConfig
from .utils import free_memory, log_gpu_memory, timer

logger = logging.getLogger(__name__)



# W&B initialisation


def init_wandb_dpo(cfg: DPOConfig) -> None:
    if "wandb" not in cfg.training.report_to:
        return
    try:
        import wandb
        wandb.init(
            project=cfg.experiment.project_name,
            name=cfg.training.run_name,
            tags=cfg.experiment.tags,
            notes=cfg.experiment.notes,
            config=cfg.to_dict(),
            resume="allow",
        )
        logger.info("W&B DPO run: %s", wandb.run.url)
    except ImportError:
        logger.warning("wandb not installed — skipping experiment tracking.")
    except Exception as exc:
        logger.warning("W&B init failed (%s) — continuing without tracking.", exc)



# DPO training arguments


def build_dpo_training_args(cfg: DPOConfig) -> TRLDPOConfig:
    tc = cfg.training
    dc = cfg.dpo

    return TRLDPOConfig(
        # Core training
        output_dir=tc.output_dir,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.learning_rate,
        num_train_epochs=tc.num_train_epochs,
        warmup_ratio=tc.warmup_ratio,
        lr_scheduler_type=tc.lr_scheduler_type,
        optim=tc.optim,
        bf16=tc.bf16,
        tf32=tc.tf32,
        # Checkpointing
        logging_steps=tc.logging_steps,
        eval_steps=tc.eval_steps,
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        load_best_model_at_end=tc.load_best_model_at_end,
        metric_for_best_model=tc.metric_for_best_model,
        greater_is_better=tc.greater_is_better,
        evaluation_strategy="steps",
        save_strategy="steps",
        # Data
        dataloader_num_workers=tc.dataloader_num_workers,
        remove_unused_columns=False,
        # DPO-specific
        beta=dc.beta,
        loss_type=dc.loss_type,
        label_smoothing=dc.label_smoothing,
        reference_free=dc.reference_free,
        max_length=cfg.data.max_seq_length,
        max_prompt_length=cfg.data.max_prompt_length,
        # Tracking
        report_to=tc.report_to,
        run_name=tc.run_name,
        seed=tc.seed,
        logging_dir=os.path.join(tc.output_dir, "logs"),
        fp16=False,
    )

# Trainer factory

def build_dpo_trainer(
    cfg: DPOConfig,
    policy_model,
    ref_model,          # None when use_peft_ref_model=True
    tokenizer,
    train_dataset,
    eval_dataset,
) -> DPOTrainer:
    dpo_args = build_dpo_training_args(cfg)

    callbacks = [
        GpuMemoryCallback(),
        EarlyStoppingCallback(
            patience=5,
            greater_is_better=cfg.training.greater_is_better,
        ),
    ]

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,      # None = use PEFT base as reference
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    return trainer


# Post-training helpers

def merge_dpo_adapter(
    trainer: DPOTrainer,
    cfg: DPOConfig,
    tokenizer,
) -> str:
    """Save adapter, reload base in bf16, merge, save full model."""
    adapter_path = os.path.join(cfg.training.output_dir, "final-adapter")
    merged_path  = os.path.join(cfg.training.output_dir, "final-merged")

    logger.info("Saving DPO LoRA adapter to %s …", adapter_path)
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    if not cfg.experiment.merge_adapter_after_training:
        logger.info("Skipping merge (merge_adapter_after_training=False).")
        return adapter_path

    logger.info("Merging DPO adapter into base model …")
    free_memory()

    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.sft_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=cfg.model.trust_remote_code,
    )
    merged = PeftModel.from_pretrained(base, adapter_path)
    merged = merged.merge_and_unload()

    logger.info("Saving merged DPO model to %s …", merged_path)
    merged.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)

    log_gpu_memory(logger, prefix="After DPO merge — ")
    return merged_path


def push_to_hub_dpo(path: str, cfg: DPOConfig, tokenizer) -> None:
    if not cfg.experiment.push_to_hub:
        return
    if not cfg.experiment.hub_model_id:
        logger.warning("push_to_hub=True but hub_model_id not set. Skipping.")
        return
    from transformers import AutoModelForCausalLM
    logger.info("Pushing DPO model to Hub: %s …", cfg.experiment.hub_model_id)
    model = AutoModelForCausalLM.from_pretrained(path)
    model.push_to_hub(cfg.experiment.hub_model_id)
    tokenizer.push_to_hub(cfg.experiment.hub_model_id)
    logger.info("DPO model pushed to Hub.")



# Main DPO entry point


def run_dpo_training(
    cfg: DPOConfig,
    policy_model,
    ref_model,
    tokenizer,
    train_dataset,
    eval_dataset,
) -> None:
    """
    Full DPO run: W&B init → train → save adapter → merge → push.
    """
    init_wandb_dpo(cfg)

    cfg.ensure_output_dirs()
    cfg.save(os.path.join(cfg.training.output_dir, "resolved_dpo_config.yaml"))

    trainer = build_dpo_trainer(
        cfg=cfg,
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    resume = cfg.training.resume_from_checkpoint
    with timer("DPO training", logger):
        logger.info(
            "Starting DPO training%s …",
            f" (resuming from {resume})" if resume else "",
        )
        trainer.train(resume_from_checkpoint=resume)

    best = trainer.state.best_metric
    logger.info(
        "DPO training complete. Best %s = %.4f",
        cfg.training.metric_for_best_model,
        best or float("nan"),
    )

    with timer("DPO merge & save", logger):
        final_path = merge_dpo_adapter(trainer, cfg, tokenizer)

    push_to_hub_dpo(final_path, cfg, tokenizer)

    try:
        import wandb
        if wandb.run:
            wandb.finish()
    except ImportError:
        pass

    logger.info("DPO complete. Final model at: %s", final_path)
