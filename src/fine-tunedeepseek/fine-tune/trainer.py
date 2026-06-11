"""
Training orchestrator.

Responsibilities
----------------
* Initialise Weights & Biases
* Build the SFTTrainer with all custom callbacks
* Run training (with optional resume)
* Post-training: merge LoRA weights → base model, optionally push to Hub
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import TrainingArguments
from trl import SFTTrainer

from .callbacks import (
    EarlyStoppingCallback,
    GpuMemoryCallback,
    SampleGenerationCallback,
)
from .config import SFTConfig
from .utils import free_memory, log_gpu_memory, timer

logger = logging.getLogger(__name__)


# W&B initialisation

def init_wandb(cfg: SFTConfig) -> None:
    """Initialise W&B if it is installed and report_to includes 'wandb'."""
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
        logger.info("W&B run initialised: %s", wandb.run.url)
    except ImportError:
        logger.warning("wandb not installed — skipping experiment tracking.")
    except Exception as exc:
        logger.warning("W&B init failed (%s) — continuing without tracking.", exc)


# TrainingArguments factory

def build_training_args(cfg: SFTConfig) -> TrainingArguments:
    tc = cfg.training
    return TrainingArguments(
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
        logging_steps=tc.logging_steps,
        eval_steps=tc.eval_steps,
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        load_best_model_at_end=tc.load_best_model_at_end,
        metric_for_best_model=tc.metric_for_best_model,
        greater_is_better=tc.greater_is_better,
        evaluation_strategy="steps",
        save_strategy="steps",
        dataloader_num_workers=tc.dataloader_num_workers,
        dataloader_pin_memory=tc.dataloader_pin_memory,
        group_by_length=tc.group_by_length,
        report_to=tc.report_to,
        run_name=tc.run_name,
        seed=tc.seed,
        logging_dir=os.path.join(tc.output_dir, "logs"),
        # Keep these off — they can cause issues with 4-bit models
        fp16=False,
        remove_unused_columns=False,
    )


# Trainer factory

def build_trainer(
    cfg: SFTConfig,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    data_collator=None,
) -> SFTTrainer:
    training_args = build_training_args(cfg)

    callbacks = [
        GpuMemoryCallback(),
        EarlyStoppingCallback(
            patience=5,
            greater_is_better=cfg.training.greater_is_better,
        ),
        SampleGenerationCallback(
            tokenizer=tokenizer,
            every_n_evals=2,
            max_new_tokens=256,
        ),
    ]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.data.max_seq_length,
        packing=cfg.data.pack_sequences,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    return trainer


# Post-training helpers

def merge_and_save(
    trainer: SFTTrainer,
    cfg: SFTConfig,
    tokenizer,
) -> str:
    """
    Merge LoRA adapter weights into the base model and save a standalone
    checkpoint.  Returns the path to the merged model.
    """
    adapter_path = os.path.join(cfg.training.output_dir, "final-adapter")
    merged_path  = os.path.join(cfg.training.output_dir, "final-merged")

    logger.info("Saving LoRA adapter to %s …", adapter_path)
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    if not cfg.experiment.merge_adapter_after_training:
        logger.info("Skipping adapter merge (merge_adapter_after_training=False).")
        return adapter_path

    logger.info("Merging adapter weights into base model …")
    free_memory()

    # Reload the base model in 16-bit for merging (4-bit models cannot be merged)
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=cfg.model.trust_remote_code,
    )
    merged = PeftModel.from_pretrained(base, adapter_path)
    merged = merged.merge_and_unload()

    logger.info("Saving merged model to %s …", merged_path)
    merged.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)

    log_gpu_memory(logger, prefix="After merge — ")
    return merged_path


def push_to_hub(path: str, cfg: SFTConfig, tokenizer) -> None:
    if not cfg.experiment.push_to_hub:
        return
    if not cfg.experiment.hub_model_id:
        logger.warning("push_to_hub=True but hub_model_id is not set. Skipping.")
        return
    logger.info("Pushing to Hub: %s …", cfg.experiment.hub_model_id)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path)
    model.push_to_hub(cfg.experiment.hub_model_id)
    tokenizer.push_to_hub(cfg.experiment.hub_model_id)
    logger.info("Model pushed to Hub successfully.")


# Main training entry point

def run_training(
    cfg: SFTConfig,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    data_collator=None,
) -> None:
    """
    Full training run:  init W&B → train → save adapter → merge → push.
    """
    init_wandb(cfg)

    # Persist the resolved config next to checkpoints
    cfg.ensure_output_dirs()
    cfg.save(os.path.join(cfg.training.output_dir, "resolved_config.yaml"))

    trainer = build_trainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    resume = cfg.training.resume_from_checkpoint
    with timer("training", logger):
        logger.info("Starting training%s …", f" (resuming from {resume})" if resume else "")
        trainer.train(resume_from_checkpoint=resume)

    logger.info("Training complete. Best eval loss: %.4f", trainer.state.best_metric or float("nan"))

    with timer("merge & save", logger):
        final_path = merge_and_save(trainer, cfg, tokenizer)

    push_to_hub(final_path, cfg, tokenizer)

    try:
        import wandb
        if wandb.run:
            wandb.finish()
    except ImportError:
        pass

    logger.info("All done. Final model at: %s", final_path)
