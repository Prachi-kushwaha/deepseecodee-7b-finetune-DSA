"""
Configuration dataclasses for the SFT pipeline.
Supports YAML loading, CLI overrides, and field validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional

import yaml

# Sub-configs

@dataclass
class ModelConfig:
    model_name: str = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    use_flash_attention_2: bool = True
    trust_remote_code: bool = True
    gradient_checkpointing: bool = True

    def __post_init__(self):
        if self.bnb_4bit_quant_type not in ("nf4", "fp4"):
            raise ValueError(f"Invalid quant_type: {self.bnb_4bit_quant_type}")
        if self.bnb_4bit_compute_dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError(f"Invalid compute_dtype: {self.bnb_4bit_compute_dtype}")


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.r <= 0 or (self.r & (self.r - 1)) != 0:
            raise ValueError(f"LoRA r must be a positive power of 2, got {self.r}")
        if self.bias not in ("none", "all", "lora_only"):
            raise ValueError(f"Invalid bias: {self.bias}")


@dataclass
class DataConfig:
    dataset_name: str = "Prachi01/dsa-dataset-unified"
    max_seq_length: int = 2048
    val_split_ratio: float = 0.05
    num_proc: int = 4
    pack_sequences: bool = True
    min_token_length: int = 32
    max_token_length: int = 2048

    def __post_init__(self):
        if not 0.0 < self.val_split_ratio < 0.5:
            raise ValueError(f"val_split_ratio must be in (0, 0.5), got {self.val_split_ratio}")
        if self.min_token_length >= self.max_token_length:
            raise ValueError("min_token_length must be less than max_token_length")


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs/deepseek-dsa-sft"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    tf32: bool = True
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    group_by_length: bool = True
    report_to: str = "wandb"
    run_name: str = "deepseek-dsa-sft-v1"
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None


@dataclass
class ExperimentConfig:
    project_name: str = "deepseek-dsa-sft"
    tags: List[str] = field(default_factory=lambda: ["deepseek", "dsa", "sft", "qlora"])
    notes: str = ""
    merge_adapter_after_training: bool = True
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

# Root config

@dataclass
class SFTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # Factory methods

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SFTConfig":
        """Load config from a YAML file."""
        with open(path) as f:
            raw: dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**raw.get("model", {})),
            lora=LoRAConfig(**raw.get("lora", {})),
            data=DataConfig(**raw.get("data", {})),
            training=TrainingConfig(**raw.get("training", {})),
            experiment=ExperimentConfig(**raw.get("experiment", {})),
        )

    def apply_cli_overrides(self, overrides: dict) -> None:
        """
        Apply flat dot-notation overrides from CLI, e.g.
            {"training.learning_rate": 1e-4, "lora.r": 32}
        """
        for key, value in overrides.items():
            section, _, attr = key.partition(".")
            sub = getattr(self, section, None)
            if sub is None or not attr:
                raise KeyError(f"Unknown config key: {key!r}")
            if not hasattr(sub, attr):
                raise KeyError(f"Unknown attribute {attr!r} in config section {section!r}")
            setattr(sub, attr, value)

    def to_dict(self) -> dict:
        """Serialize full config to a plain dict (useful for W&B logging)."""
        import dataclasses
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        """Persist resolved config back to YAML."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def ensure_output_dirs(self) -> None:
        os.makedirs(self.training.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.training.output_dir, "logs"), exist_ok=True)
