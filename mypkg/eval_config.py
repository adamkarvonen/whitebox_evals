from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional
from dataclasses import field

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict


class InferenceMode(str, Enum):
    GPU_INFERENCE = "gpu_inference"
    GPU_FORWARD_PASS = "gpu_forward_pass"
    PERFORM_ABLATIONS = "perform_ablations"
    PROJECTION_ABLATIONS = "projection_ablations"
    OPEN_ROUTER = "open_router"
    LOGIT_LENS = "logit_lens"
    LOGIT_LENS_WITH_INTERVENTION = "logit_lens_with_intervention"

    def __str__(self):
        return self.value


class EvalConfig(BaseModel, extra="forbid"):
    inference_mode: InferenceMode

    random_seed: int = 42

    model_name: str = ""

    anti_bias_statement_file: str = ""
    job_description_file: str = ""
    system_prompt_filename: str = "yes_no.txt"
    anti_bias_statement_folder: str = "generated_anti_bias_statements"
    job_description_folder: str = "job_descriptions"

    email_domain: str = "gmail"
    # Note: this is currently not used and is hardcoded in dataset.py
    resume_dataset_path: str = "data/resume/selected_cats_resumes.csv"
    score_output_dir: str = "score_output"

    industry: str = "INFORMATION-TECHNOLOGY"
    anthropic_dataset: bool = False
    downsample: int | None = None
    no_names: bool = False
    batch_size_multiplier: int = 2
    max_length: int = 2500
    overwrite_existing_results: bool = False
    sae_intervention_type: str = "clamping"

    # For ablation experiments only
    scale: float = 1000.0
    bias_type: str = "N/A"

    # model_names = [
    #             "google/gemma-2-2b-it",
    #             # "google/gemma-2-27b-it",
    #             # "google/gemma-2-9b-it",
    #             # "mistralai/Ministral-8B-Instruct-2410",
    #             # "mistralai/Mistral-Small-24B-Instruct-2501",
    #             # "deepseek/deepseek-r1",
    #             # "openai/gpt-4o-2024-08-06",
    #             # "deepseek/deepseek-r1-distill-llama-70b"
    #             # "openai/o1-mini-2024-09-12",
    #             # "openai/o1-mini",
    #             # "openai/o1"
    #             # "x-ai/grok-3-mini-beta"
    #             # "qwen/qwq-32b",
    #             # "anthropic/claude-3.7-sonnet"
    #             # "anthropic/claude-3.7-sonnet:thinking",
    #             # "qwen/qwen2.5-32b-instruct",
    #             # "openai/gpt-4o-mini",
    #         ]

    model_names_to_iterate: list[str] = field(default_factory=list)
    anti_bias_statement_files_to_iterate: list[str] = field(default_factory=list)
    job_description_files_to_iterate: list[str] = field(default_factory=list)
    bias_types_to_iterate: list[str] = field(default_factory=list)
    scales_to_iterate: list[float] = field(default_factory=list)

    # ------------- convenience IO helpers -------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)  # full type check

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump()))


class FrozenEvalConfig(EvalConfig):
    model_config = ConfigDict(frozen=True)
