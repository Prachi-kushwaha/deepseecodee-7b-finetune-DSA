"""
Configuration dataclasses for the DPO pipeline.
Extends the existing SFT config pattern.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import yaml


# Sub-configs (DPO-specific)

@dataclass
class DPOModelConfig:
    sft_model_path: str = "./outputs/deepseek-dsa-sft/final-merged"
    base_model_name: str = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    use_flash_attention_2: bool = True
    trust_remote_code: bool = True
    gradient_checkpointing: bool = True
    # Memory-efficient: disable LoRA adapters for the reference forward pass
    # instead of loading a second copy of the model
    use_peft_ref_model: bool = True

    def __post_init__(self):
        if not self.sft_model_path:
            raise ValueError("sft_model_path must be set")


@dataclass
class DPOLoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DPODataConfig:
    preference_dataset_path: str = ""
    preference_dataset_name: str = ""
    generate_synthetic_preferences: bool = True
    source_dataset_name: str = "Prachi01/dsa-dataset-unified"
    num_samples_for_generation: int = 500
    num_generations_per_prompt: int = 4
    max_seq_length: int = 1024
    max_prompt_length: int = 512
    val_split_ratio: float = 0.05
    num_proc: int = 4

    def __post_init__(self):
        if (
            not self.preference_dataset_path
            and not self.preference_dataset_name
            and not self.generate_synthetic_preferences
        ):
            raise ValueError(
                "Must provide preference_dataset_path, preference_dataset_name, "
                "or set generate_synthetic_preferences=True"
            )


@dataclass
class DPOAlgorithmConfig:
    beta: float = 0.1
    loss_type: Literal["sigmoid", "ipo", "hinge", "robust"] = "sigmoid"
    label_smoothing: float = 0.0
    reference_free: bool = False

    def __post_init__(self):
        if not 0.0 < self.beta < 1.0:
            raise ValueError(f"beta must be in (0, 1), got {self.beta}")
        if self.loss_type not in ("sigmoid", "ipo", "hinge", "robust"):
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


@dataclass
class DPOTrainingConfig:
    output_dir: str = "./outputs/deepseek-dsa-dpo"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    num_train_epochs: int = 1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    tf32: bool = True
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_rewards/margins"
    greater_is_better: bool = True
    dataloader_num_workers: int = 2
    report_to: str = "wandb"
    run_name: str = "deepseek-dsa-dpo-v1"
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None


@dataclass
class ScoringConfig:
    strategy: Literal["execution", "length_penalty", "judge_llm"] = "execution"
    execution_timeout_seconds: int = 5
    judge_model: str = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"


@dataclass
class DPOExperimentConfig:
    project_name: str = "deepseek-dsa-dpo"
    tags: List[str] = field(
        default_factory=lambda: ["deepseek", "dsa", "dpo", "qlora"]
    )
    notes: str = ""
    merge_adapter_after_training: bool = True
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Root DPO config
# ---------------------------------------------------------------------------

@dataclass
class DPOConfig:
    model: DPOModelConfig = field(default_factory=DPOModelConfig)
    lora: DPOLoRAConfig = field(default_factory=DPOLoRAConfig)
    data: DPODataConfig = field(default_factory=DPODataConfig)
    dpo: DPOAlgorithmConfig = field(default_factory=DPOAlgorithmConfig)
    training: DPOTrainingConfig = field(default_factory=DPOTrainingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    experiment: DPOExperimentConfig = field(default_factory=DPOExperimentConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DPOConfig":
        with open(path) as f:
            raw: dict = yaml.safe_load(f)
        return cls(
            model=DPOModelConfig(**raw.get("model", {})),
            lora=DPOLoRAConfig(**raw.get("lora", {})),
            data=DPODataConfig(**raw.get("data", {})),
            dpo=DPOAlgorithmConfig(**raw.get("dpo", {})),
            training=DPOTrainingConfig(**raw.get("training", {})),
            scoring=ScoringConfig(**raw.get("scoring", {})),
            experiment=DPOExperimentConfig(**raw.get("experiment", {})),
        )

    def apply_cli_overrides(self, overrides: dict) -> None:
        for key, value in overrides.items():
            section, _, attr = key.partition(".")
            sub = getattr(self, section, None)
            if sub is None or not attr:
                raise KeyError(f"Unknown config key: {key!r}")
            setattr(sub, attr, value)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def ensure_output_dirs(self) -> None:
        os.makedirs(self.training.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.training.output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.training.output_dir, "preference_data"), exist_ok=True)
