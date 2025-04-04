from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class EvalConfig:
    random_seed: int = 42

    model_name: str = "mistralai/Ministral-8B-Instruct-2410"

    anti_bias_statement_file: str = "anti_bias_statements.txt"
    job_description_file: str = "job_description.txt"
    system_prompt_filename: str = "yes_no.txt"
    anti_bias_statement_folder: str = "generated_anti_bias_statements"
    job_description_folder: str = "job_descriptions"

    email_domain: str = "gmail"

    industry: str = "INFORMATION-TECHNOLOGY"
    mode: str = "full"
    political_orientation: bool = False
    pregnancy: bool = False
    employment_gap: bool = False
    anthropic_dataset: bool = False
    downsample: int = None
    gpu_inference: bool = False
    steering_intervention: bool = False
    ablation_intervention: bool = False
    intervention_strength: float = 0.0
    intervened_features: list[str] = field(default_factory=list)
