import torch
import einops
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import os
import random
from copy import deepcopy
from typing import Callable, Optional, Any
from tqdm import tqdm
import pickle
import gc
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from datasets import load_dataset

import mypkg.whitebox_infra.attribution as attribution
import mypkg.whitebox_infra.dictionaries.batch_topk_sae as batch_topk_sae
import mypkg.whitebox_infra.data_utils as data_utils
import mypkg.whitebox_infra.model_utils as model_utils
import mypkg.whitebox_infra.interp_utils as interp_utils
import mypkg.pipeline.setup.dataset as dataset_setup
import mypkg.pipeline.infra.hiring_bias_prompts as hiring_bias_prompts
from mypkg.eval_config import EvalConfig
import mypkg.pipeline.infra.model_inference as model_inference


def create_paired_resumes(resumes: list[str]) -> list[str]:
    new_resumes = []

    name = "Taylor Johnson"
    email = "taylor.johnson@gmail.com"

    for resume in resumes:
        resume1 = f"Name: {name} (He/him)\nEmail: {email}\n\n{resume}"
        resume2 = f"Name: {name} (She/her)\nEmail: {email}\n\n{resume}"

        new_resumes.append(resume1)
        new_resumes.append(resume2)

    return new_resumes


def load_anthropic_datasets(dataset_type="implicit", downsample_to=None):
    print("Loading dataset...")

    print(f"Loading Anthropic dataset ({dataset_type})...")
    dataset = load_dataset("Anthropic/discrim-eval", dataset_type)
    df = dataset["train"]
    # filtered_df = [item for item in df if item["decision_question_id"] == 16]
    df = pd.DataFrame(df)

    return df


def contrastive_gender_pairs(df: pd.DataFrame) -> list[str]:
    key_cols = ["decision_question_id", "race", "fill_type", "age"]

    pairs: list[str] = []

    # Group by the invariant fields
    for _, group in df.groupby(key_cols):
        # If the group has fewer than 2 distinct genders, nothing to pair
        if group["gender"].nunique() < 2:
            continue

        # Iterate over all index pairs whose genders differ
        for i, j in itertools.combinations(group.index, 2):
            if group.at[i, "gender"] != group.at[j, "gender"]:
                pairs.append(group.at[i, "filled_template"])
                pairs.append(group.at[j, "filled_template"])

    return pairs


def get_split_dataloaders(
    batch_size: int, downsample: int, train_ratio: float = 0.8, random_seed: int = 42
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    anthropic_df = load_anthropic_datasets()
    train_pairs = contrastive_gender_pairs(anthropic_df)
    train_labels = [0, 1] * (len(train_pairs) // 2)
    train_prompts = [""] * len(train_pairs)

    resume_df = dataset_setup.load_raw_dataset()

    industry = "INFORMATION-TECHNOLOGY"
    resume_df = dataset_setup.filter_by_industry(resume_df, industry)

    resume_df = dataset_setup.balanced_downsample(resume_df, downsample, random_seed)

    unique_resumes = list(resume_df["Resume_str"].unique())

    paired_resumes = create_paired_resumes(unique_resumes)

    all_labels = [0, 1] * len(unique_resumes)
    all_prompts = [""] * len(paired_resumes)

    test_resumes = paired_resumes
    test_labels = all_labels
    test_prompts = all_prompts

    # train_resumes = paired_resumes[:train_size]
    # train_labels = all_labels[:train_size]
    # train_prompts = all_prompts[:train_size]

    # test_resumes = paired_resumes[train_size:]
    # test_labels = all_labels[train_size:]
    # test_prompts = all_prompts[train_size:]

    train_dataloader = data_utils.create_simple_dataloader(
        train_pairs,
        train_labels,
        train_prompts,
        model_name,
        device,
        batch_size=batch_size,
        sort_by_length=False,
        max_length=2500,
    )

    test_dataloader = data_utils.create_simple_dataloader(
        test_resumes,
        test_labels,
        test_prompts,
        model_name,
        device,
        batch_size=batch_size,
        sort_by_length=False,
        max_length=2500,
    )

    return train_dataloader, test_dataloader


def get_dataloaders(
    batch_size: int, downsample: int, train_ratio: float = 0.8, random_seed: int = 42
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    df = dataset_setup.load_raw_dataset()

    industry = "INFORMATION-TECHNOLOGY"
    df = dataset_setup.filter_by_industry(df, industry)

    df = dataset_setup.balanced_downsample(df, downsample, random_seed)

    unique_resumes = list(df["Resume_str"].unique())

    paired_resumes = create_paired_resumes(unique_resumes)

    all_labels = [0, 1] * len(unique_resumes)
    all_prompts = [""] * len(paired_resumes)

    train_size = int(len(all_labels) * train_ratio)

    train_resumes = paired_resumes[:train_size]
    train_labels = all_labels[:train_size]
    train_prompts = all_prompts[:train_size]

    test_resumes = paired_resumes[train_size:]
    test_labels = all_labels[train_size:]
    test_prompts = all_prompts[train_size:]

    train_dataloader = data_utils.create_simple_dataloader(
        train_resumes,
        train_labels,
        train_prompts,
        model_name,
        device,
        batch_size=batch_size,
        sort_by_length=False,
        max_length=2500,
    )

    test_dataloader = data_utils.create_simple_dataloader(
        test_resumes,
        test_labels,
        test_prompts,
        model_name,
        device,
        batch_size=batch_size,
        sort_by_length=False,
        max_length=2500,
    )

    return train_dataloader, test_dataloader


def token_distance(a: torch.Tensor, b: torch.Tensor, metric: str) -> torch.Tensor:
    """
    a, b: (L, D) activation tensors for one sample each
    returns: (L,) distance for every token position
    """
    if metric == "euclidean":
        # ‑> ||a - b||₂
        return torch.linalg.norm(a - b, dim=-1)
    elif metric == "cosine":
        # 1 ‑ cos_sim so that identical vectors give 0 distance
        return 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=-1)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


class LinearProjection(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, input_dim)
        # Initialize weights and bias to zero
        self.projection.weight.data.zero_()
        self.projection.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        return self.projection(x.float()).to(orig_dtype)


class LowRankLinearProjection(torch.nn.Module):
    def __init__(self, input_dim: int, rank: int):
        super().__init__()
        self.projection1 = torch.nn.Linear(input_dim, rank)
        self.projection2 = torch.nn.Linear(rank, input_dim)
        # Initialize projection2.weight and bias to zero
        self.projection2.weight.data.zero_()
        self.projection2.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = self.projection1(x.float())
        x = self.projection2(x)
        return x.to(orig_dtype)


def get_adapter_loss(
    model: AutoModelForCausalLM,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any],
    layer: int,
    adapter: LinearProjection,
    lambda_reg: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    submodule = model_utils.get_submodule(model, layer)
    input_ids, attention_mask_BL, labels, idx_batch, resume_prompt_results_batch = batch
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask_BL,
    }

    activations_BLD = model_utils.collect_activations(
        model, submodule, model_inputs, use_no_grad=True
    )
    gender_component_BLD = adapter(activations_BLD)
    modified_activations_BLD = activations_BLD - gender_component_BLD

    B, L, D = activations_BLD.shape
    assert B % 2 == 0, "Batch must contain an even number of elements"

    assert torch.all(attention_mask_BL[0::2] == attention_mask_BL[1::2]), (
        "Attention masks must match"
    )

    # walk through paired elements 0&1, 2&3, …
    for i in range(0, B, 2):
        assert int(labels[i]) + int(labels[i + 1]) == 1, "Labels must be 0 or 1"

    a_tokens_bLD = modified_activations_BLD[0::2]
    b_tokens_bLD = modified_activations_BLD[1::2]

    half_attention_mask_BL = attention_mask_BL[0::2]

    num_valid_tokens = attention_mask_BL.sum()
    num_valid_tokens_half = half_attention_mask_BL.sum()

    assert num_valid_tokens > 0, "No valid tokens in batch"

    mse_BL = (
        torch.nn.functional.mse_loss(a_tokens_bLD, b_tokens_bLD, reduction="none").mean(
            dim=-1
        )
        * half_attention_mask_BL
    )

    mse = mse_BL.float().sum() / num_valid_tokens_half

    cosine_sim_BL = (
        torch.nn.functional.cosine_similarity(a_tokens_bLD, b_tokens_bLD, dim=-1)
        * half_attention_mask_BL
    )
    cosine_sim = cosine_sim_BL.float().sum() / num_valid_tokens_half

    similarity_loss = mse

    norm_per_token_BL = (activations_BLD**2).sum(dim=-1) * attention_mask_BL
    gender_component_per_token_BL = (gender_component_BLD**2).sum(
        dim=-1
    ) * attention_mask_BL

    norm_per_token = norm_per_token_BL.float().sum() / num_valid_tokens
    gender_component_per_token = (
        gender_component_per_token_BL.float().sum() / num_valid_tokens
    )

    regularization_loss = lambda_reg * (
        gender_component_per_token / (norm_per_token + 1e-8)
    )

    loss = similarity_loss + regularization_loss

    return loss, similarity_loss, cosine_sim, regularization_loss


@torch.no_grad()
def get_adapter_val_loss(
    model: AutoModelForCausalLM,
    dataloader: torch.utils.data.DataLoader,
    layer: int,
    adapter: LinearProjection,
    lambda_reg: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total_loss = 0.0
    total_sim_loss = 0.0
    total_cos_sim = 0.0
    total_reg_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        loss, sim_loss, cos_sim, reg_loss = get_adapter_loss(
            model, batch, layer, adapter, lambda_reg
        )
        total_loss += loss.item()
        total_sim_loss += sim_loss.item()
        total_cos_sim += cos_sim.item()
        total_reg_loss += reg_loss.item()
        n_batches += 1

    # turn the Python floats back into 0‑D tensors so the caller
    # receives the same type it got from get_adapter_loss
    mean_loss = torch.tensor(total_loss / n_batches)
    mean_sim = torch.tensor(total_sim_loss / n_batches)
    mean_cos_sim = torch.tensor(total_cos_sim / n_batches)
    mean_reg = torch.tensor(total_reg_loss / n_batches)

    return mean_loss, mean_sim, mean_cos_sim, mean_reg


def train_adapter(
    model: AutoModelForCausalLM,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    layer: int,
    distance_metric: str,
    device: torch.device,
    lr: float = 1e-3,
    num_epochs: int = 2,
) -> None:
    hidden_dim = model.config.hidden_size

    # adapter = LinearProjection(hidden_dim).to(device)

    adapter = LowRankLinearProjection(hidden_dim, 16).to(device)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.01)
    lambda_reg = 1.0

    with torch.no_grad():
        val_loss, val_similarity_loss, val_cosine_sim, val_regularization_loss = (
            get_adapter_val_loss(model, test_dataloader, layer, adapter, lambda_reg)
        )
        print(
            f"Initial val loss: {val_loss}, Similarity Loss: {val_similarity_loss}, Cosine Sim: {val_cosine_sim}, Regularization Loss: {val_regularization_loss}"
        )

    for epoch in range(num_epochs):
        for i, batch in enumerate(train_dataloader):
            loss, similarity_loss, cosine_sim, regularization_loss = get_adapter_loss(
                model, batch, layer, adapter, lambda_reg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(
            #     f"Epoch {epoch}, Batch {i}, Loss: {loss}, Similarity Loss: {similarity_loss}, Cosine Sim: {cosine_sim}, Regularization Loss: {regularization_loss}"
            # )

        with torch.no_grad():
            val_loss, val_similarity_loss, val_cosine_sim, val_regularization_loss = (
                get_adapter_val_loss(model, test_dataloader, layer, adapter, lambda_reg)
            )
            print(
                f"Epoch {epoch}, Validation loss: {val_loss}, Similarity Loss: {val_similarity_loss}, Cosine Sim: {val_cosine_sim}, Regularization Loss: {val_regularization_loss}"
            )

    return adapter


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "google/gemma-2-2b-it"

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    model.requires_grad_(False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    batch_size = 4

    assert batch_size % 2 == 0

    DISTANCE_METRIC = "euclidean"
    DISTANCE_METRIC = "cosine"

    layer = 0

    train_dataloader, test_dataloader = get_dataloaders(
        batch_size, downsample=150, train_ratio=0.8, random_seed=random_seed
    )

    model = model_utils.truncate_model(model, layer)

    adapter = train_adapter(
        model,
        train_dataloader,
        test_dataloader,
        layer,
        DISTANCE_METRIC,
        device,
        num_epochs=10,
    )
