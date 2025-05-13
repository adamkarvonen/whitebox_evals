# %%

%load_ext autoreload
%autoreload 2

# %%

import torch
import einops
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import os
import random
from copy import deepcopy
from typing import Callable, Optional
from tqdm import tqdm
import pickle
import gc
import pandas as pd
import matplotlib.pyplot as plt

import mypkg.whitebox_infra.attribution as attribution
import mypkg.whitebox_infra.dictionaries.batch_topk_sae as batch_topk_sae
import mypkg.whitebox_infra.data_utils as data_utils
import mypkg.whitebox_infra.model_utils as model_utils
import mypkg.whitebox_infra.interp_utils as interp_utils
import mypkg.pipeline.setup.dataset as dataset_setup
import mypkg.pipeline.infra.hiring_bias_prompts as hiring_bias_prompts
from mypkg.eval_config import EvalConfig
import mypkg.pipeline.infra.model_inference as model_inference

# %%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = "google/gemma-2-2b-it"

dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=dtype, device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%

batch_size = 4

assert batch_size % 2 == 0

# %%

df = dataset_setup.load_raw_dataset()

industry = "INFORMATION-TECHNOLOGY"
downsample = 10
random_seed = 42
df = dataset_setup.filter_by_industry(df, industry)

df = dataset_setup.balanced_downsample(df, downsample, random_seed)

# %%

unique_resumes = list(df["Resume_str"].unique())
print(type(unique_resumes))
print(len(unique_resumes))

# %%

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

paired_resumes = create_paired_resumes(unique_resumes)

print(len(paired_resumes))

train_labels = [0,1] * len(unique_resumes)
train_prompts = [""] * len(paired_resumes)
print(len(train_labels))

# %%

dataloader = data_utils.create_simple_dataloader(
    paired_resumes, train_labels, train_prompts, model_name, device, batch_size=batch_size, sort_by_length=False, max_length=2500
)

# %%

# for batch in dataloader:
#     input_ids, attention_mask, labels, idx_batch, resume_prompt_results_batch = batch
#     print(input_ids.shape)
#     print(attention_mask.shape)
#     attention_mask_B = attention_mask.sum(dim=1)
#     print(attention_mask_B)

# %%

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

# choose one of {"euclidean", "cosine"}
DISTANCE_METRIC = "euclidean"
DISTANCE_METRIC = "cosine"

layer = 5


submodule = model_utils.get_submodule(model, layer)

all_pair_dists = [] 
for batch in dataloader:
    input_ids, attention_mask, labels, idx_batch, resume_prompt_results_batch = batch

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    activations_BLD = model_utils.collect_activations(model, submodule, model_inputs)
    print(activations_BLD.shape)

    B, L, D = activations_BLD.shape
    assert B % 2 == 0, "Batch must contain an even number of elements"

    # walk through paired elements 0&1, 2&3, …
    for i in range(0, B, 2):
        # sanity check: tokenization lengths must match inside each pair
        assert torch.equal(attention_mask[i], attention_mask[i + 1])

        # mask out padding / unused tokens
        valid_token_mask = attention_mask[i].bool()          # shape (L,)

        a_tok = activations_BLD[i][valid_token_mask]         # (L_valid, D)
        b_tok = activations_BLD[i + 1][valid_token_mask]     # (L_valid, D)

        dists_L = token_distance(a_tok, b_tok, DISTANCE_METRIC)  # (L_valid,)
        all_pair_dists.append(dists_L)

    
    # break

# %%

for i in range(len(all_pair_dists)):
    all_pair_dists[i].to("cpu").float()


# %%

pair_dists_padded = torch.nn.utils.rnn.pad_sequence(
    all_pair_dists,
    batch_first=True,
    padding_value=float("nan"),
).to("cpu").float()

# optional: cache to disk so you can iterate on plotting without recomputing
torch.save(pair_dists_padded, "paired_distances.pt")
print("Saved distance tensor with shape", pair_dists_padded.shape)

# ---- quick‑n‑dirty histogram -------------------------------------------------

flat_dists = pair_dists_padded.flatten()
flat_dists = flat_dists[~torch.isnan(flat_dists)]   # drop NaNs

plt.hist(flat_dists.numpy(), bins=100)
plt.title(f"{DISTANCE_METRIC.capitalize()} distance per token")
plt.xlabel("Distance")
plt.ylabel("Count")
plt.yscale("log")
plt.tight_layout()
plt.show()

# %%

import torch
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# load the NaN‑padded tensor we saved earlier
pair_dists_padded = torch.load("paired_distances.pt")      # (N_pairs, max_len)
metric_name = "cosine"   # or "euclidean" – keep in sync with the file you saved
# --------------------------------------------------------------

# ------- 2A. raw heat‑map (pair × token position) -------------
plt.figure(figsize=(14, 6))
plt.imshow(
    pair_dists_padded,        # NaNs render as blanched / white
    aspect="auto",
    interpolation="nearest",
    origin="lower"
)
plt.colorbar(label=f"{metric_name} distance")
plt.xlabel("Token position")
plt.ylabel("Resume pair index")
plt.title(f"Per‑token {metric_name} distance heat‑map")
plt.tight_layout()
plt.show()

# ------- 2B. average across all pairs --------------------------
mean_dist = torch.nanmean(pair_dists_padded, dim=0)        # (max_len,)
plt.figure(figsize=(12, 4))
plt.plot(mean_dist.numpy())
plt.xlabel("Token position")
plt.ylabel(f"Mean {metric_name} distance")
plt.title(f"Average per‑token {metric_name} distance")
plt.tight_layout()
plt.show()

# %%