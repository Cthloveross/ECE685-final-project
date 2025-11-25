from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]

DatasetName = Literal["nq_open", "real_toxicity_prompts", "anthropic_hh"]


@dataclass(slots=True)
class DataPaths:
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    tmp_dir: Path = ROOT / "data" / "tmp"
    results_dir: Path = ROOT / "data" / "results"

    def dataset_file(self, dataset: DatasetName) -> Path:
        return self.raw_dir / f"{dataset}.jsonl"


@dataclass(slots=True)
class ModelConfig:
    gemma_model_name: str = "google/gemma-2b-it"
    toxicity_model_name: str = "unitary/unbiased-toxic-roberta"
    hook_layer: int = 12  # Layer to hook for SAE (Gemma-2B has 18 layers: 0-17)
    max_length: int = 512
    dtype: str = "bfloat16"
    
    # Gemma Scope SAE configuration
    sae_release: str = "gemma-scope-2b-pt-res"
    sae_id: str = "layer_12/width_16k/average_l0_71"  # 16k features, layer 12


@dataclass(slots=True)
class ExperimentConfig:
    device: str = "cuda"
    seed: int = 42
    steering_strengths: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)


@dataclass(slots=True)
class ProjectConfig:
    data: DataPaths = field(default_factory=DataPaths)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


CONFIG = ProjectConfig()
