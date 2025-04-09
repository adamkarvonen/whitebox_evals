# prompt_utils.py

import os
import torch
import einops
from tqdm import tqdm
from typing import Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from mypkg.whitebox_infra.dictionaries import topk_sae, base_sae
from mypkg.whitebox_infra.model_utils import collect_activations
from mypkg.whitebox_infra.data_utils import load_and_tokenize_and_concat_dataset


@torch.no_grad()
def get_max_activating_prompts(
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    tokenized_inputs_bL: list[dict[str, torch.Tensor]],
    dim_indices: torch.Tensor,
    batch_size: int,
    dictionary: base_sae.BaseSAE,
    context_length: int,
    k: int = 30,
    zero_bos: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each feature in dim_indices, find the top-k (prompt, position) with the highest
    dictionary-encoded activation. Return the tokens and the activations for those points.
    """

    device = model.device
    feature_count = dim_indices.shape[0]

    # We'll store results in [F, k] or [F, k, L] shape
    max_activating_indices_FK = torch.zeros(
        (feature_count, k), device=device, dtype=torch.int32
    )
    max_activations_FK = torch.zeros(
        (feature_count, k), device=device, dtype=torch.bfloat16
    )
    max_tokens_FKL = torch.zeros(
        (feature_count, k, context_length), device=device, dtype=torch.int32
    )
    max_activations_FKL = torch.zeros(
        (feature_count, k, context_length), device=device, dtype=torch.bfloat16
    )

    for i, inputs_BL in tqdm(
        enumerate(tokenized_inputs_bL), total=len(tokenized_inputs_bL)
    ):
        batch_offset = i * batch_size
        attention_mask = inputs_BL["attention_mask"]

        # 1) Collect submodule activations
        activations_BLD = collect_activations(model, submodule, inputs_BL)

        # 2) Apply dictionary's encoder
        #    shape: [B, L, D], dictionary.encode -> [B, L, F]
        #    Then keep only the dims in dim_indices
        activations_BLF = dictionary.encode(activations_BLD)
        if zero_bos:
            # Zero out BOS token if you want to ignore activations at position 0
            activations_BLF[:, 0, :] = 0.0

        activations_BLF = activations_BLF[:, :, dim_indices]  # shape: [B, L, Fselected]

        activations_BLF = activations_BLF * attention_mask[:, :, None]

        # 3) Move dimension to (F, B, L)
        activations_FBL = einops.rearrange(activations_BLF, "B L F -> F B L")

        # For each sequence, the "peak activation" is the maximum over positions:
        # shape: [F, B]
        activations_FB = einops.reduce(activations_FBL, "F B L -> F B", "max")

        # We'll replicate the tokens to shape [F, B, L]
        tokens_FBL = einops.repeat(
            inputs_BL["input_ids"], "B L -> F B L", F=feature_count
        )

        # Create an index for the batch offset
        indices_B = torch.arange(batch_offset, batch_offset + batch_size, device=device)
        indices_FB = einops.repeat(indices_B, "B -> F B", F=feature_count)

        # Concatenate with previous top-k
        combined_activations_FB = torch.cat([max_activations_FK, activations_FB], dim=1)
        combined_indices_FB = torch.cat([max_activating_indices_FK, indices_FB], dim=1)

        combined_activations_FBL = torch.cat(
            [max_activations_FKL, activations_FBL], dim=1
        )
        combined_tokens_FBL = torch.cat([max_tokens_FKL, tokens_FBL], dim=1)

        # 4) Sort to keep only top-k
        topk_activations_FK, topk_indices_FK = torch.topk(
            combined_activations_FB, k, dim=1
        )

        max_activations_FK = topk_activations_FK
        feature_indices_F1 = torch.arange(feature_count, device=device)[:, None]

        max_activating_indices_FK = combined_indices_FB[
            feature_indices_F1, topk_indices_FK
        ]
        max_activations_FKL = combined_activations_FBL[
            feature_indices_F1, topk_indices_FK
        ]
        max_tokens_FKL = combined_tokens_FBL[feature_indices_F1, topk_indices_FK]

    return max_tokens_FKL, max_activations_FKL


@torch.no_grad()
def get_all_prompts_activations(
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    tokenized_inputs_bL: list[dict[str, torch.Tensor]],
    dim_indices: torch.Tensor,
    dictionary: base_sae.BaseSAE,
    zero_bos: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ """
    device = model.device
    all_tokens_list = []
    all_activations_list = []

    # Ensure dim_indices is on the correct device
    dim_indices = dim_indices.to(device)

    for inputs_BL in tqdm(tokenized_inputs_bL, desc="Processing batches"):
        # Ensure batch is on the correct device
        input_ids_BL = inputs_BL["input_ids"].to(device)
        attention_mask_BL = inputs_BL["attention_mask"].to(device)

        # 1) Collect submodule activations
        activations_BLD = collect_activations(
            model,
            submodule,
            {"input_ids": input_ids_BL, "attention_mask": attention_mask_BL},
        )

        # 2) Apply dictionary's encoder
        #    shape: [B, L, D_dict], dictionary.encode -> [B, L, F_dict]
        activations_BLF = dictionary.encode(activations_BLD)

        # 3) Select desired features
        #    Ensure dim_indices is compatible with the last dimension of activations_BLF_full
        activations_BLF = activations_BLF[:, :, dim_indices]

        # 4) Optional: Zero out BOS token activations
        if zero_bos:
            activations_BLF[:, 0, :] = 0.0

        # 5) Apply attention mask
        activations_BLF = activations_BLF * attention_mask_BL[:, :, None]

        # Append results for this batch (move back to CPU to avoid accumulating on GPU)
        all_tokens_list.append(input_ids_BL.cpu())
        all_activations_list.append(activations_BLF.cpu())

    # Concatenate results from all batches
    all_tokens_NL = torch.cat(all_tokens_list, dim=0)
    all_tokens_FBL = einops.repeat(all_tokens_NL, "B L -> F B L", F=len(dim_indices))
    # Concatenate results from all batches
    all_activations_BLF = torch.cat(all_activations_list, dim=0)

    all_activations_FBL = einops.rearrange(all_activations_BLF, "B L F -> F B L")

    return all_tokens_FBL, all_activations_FBL


# ================================
# Main user-facing function
# ================================


def get_interp_prompts(
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    sae: base_sae.BaseSAE,
    dim_indices: torch.Tensor,
    context_length: int,
    tokenizer: AutoTokenizer,
    dataset_name: str = "togethercomputer/RedPajama-Data-V2",
    num_tokens: int = 1_000_000,
    batch_size: int = 32,
    tokens_folder: str = "tokens",
    force_rebuild_tokens: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    1) Loads or builds a tokenized dataset (B, context_length).
    2) Splits into batches of size batch_size.
    3) Runs get_max_activating_prompts(...) to get top-k tokens/activations.

    :param model: A loaded AutoModelForCausalLM (already on device).
    :param submodule: The submodule (e.g. layer) on which to hook activations.
    :param sae: Your dictionary-learning topk model (sae).
    :param dim_indices: A LongTensor of dimension indices of interest [F].
    :param context_length: The sequence length used in the model.
    :param tokenizer: The tokenizer corresponding to your model.
    :param dataset_name: Name for the HF dataset from which to load text.
    :param num_tokens: How many tokens total we want (roughly).
    :param batch_size: Batch size for forward passes.
    :param tokens_folder: Where to cache the tokenized dataset.
    :param force_rebuild_tokens: If True, ignore any existing token cache and rebuild.
    :return: (max_tokens_FKL, max_activations_FKL)
    """
    device = model.device
    model_name = getattr(model.config, "_name_or_path", "unknown_model")

    # E.g. "tokens/togethercomputer_RedPajama-Data-V2_1000000_google-gemma-2-2b.pt"
    filename = f"{tokens_folder}/{dataset_name.replace('/', '_')}_{num_tokens}_{model_name.replace('/', '_')}.pt"
    os.makedirs(tokens_folder, exist_ok=True)

    # If we haven't built the token file or if user wants to force a rebuild
    if (not os.path.exists(filename)) or force_rebuild_tokens:
        token_dict = load_and_tokenize_and_concat_dataset(
            dataset_name=dataset_name,
            ctx_len=context_length,
            num_tokens=num_tokens,
            tokenizer=tokenizer,
            add_bos=True,
        )
        token_dict = {k: v.cpu() for k, v in token_dict.items()}
        torch.save(token_dict, filename)
        print(f"Saved tokenized dataset to {filename}")
    else:
        print(f"Loading tokenized dataset from {filename}")
        token_dict = torch.load(filename)

    token_dict = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in token_dict.items()
    }

    batched_tokens = []

    for i in range(0, token_dict["input_ids"].shape[0], batch_size):
        batched_tokens.append(
            {
                "input_ids": token_dict["input_ids"][i : i + batch_size],
                "attention_mask": token_dict["attention_mask"][i : i + batch_size],
            }
        )

    # Now get the max-activating prompts for the given dim_indices
    max_tokens_FKL, max_activations_FKL = get_max_activating_prompts(
        model=model,
        submodule=submodule,
        tokenized_inputs_bL=batched_tokens,
        dim_indices=dim_indices,
        batch_size=batch_size,
        dictionary=sae,
        context_length=context_length,
        k=30,  # or pass as a parameter if you want
    )

    return max_tokens_FKL, max_activations_FKL


def get_interp_prompts_user_inputs(
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    sae: base_sae.BaseSAE,
    dim_indices: torch.Tensor,
    user_inputs: list[str],
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    k: int = 30,
    sort_by_activation: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = model.device

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset_tokens = tokenizer(
        user_inputs,
        return_tensors="pt",
        padding=True,
        padding_side="right",
        add_special_tokens=True,
    )

    dataset_tokens = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in dataset_tokens.items()
    }

    seq_length = dataset_tokens["input_ids"].shape[1]

    batched_tokens = []

    for i in range(0, dataset_tokens["input_ids"].shape[0], batch_size):
        batch_tokens = {
            "input_ids": dataset_tokens["input_ids"][i : i + batch_size],
            "attention_mask": dataset_tokens["attention_mask"][i : i + batch_size],
        }
        batched_tokens.append(batch_tokens)

    if sort_by_activation:
        # Now get the max-activating prompts for the given dim_indices
        max_tokens_FKL, max_activations_FKL = get_max_activating_prompts(
            model=model,
            submodule=submodule,
            tokenized_inputs_bL=batched_tokens,
            dim_indices=dim_indices,
            batch_size=batch_size,
            dictionary=sae,
            context_length=seq_length,
            k=k,
        )
    else:
        max_tokens_FKL, max_activations_FKL = get_all_prompts_activations(
            model=model,
            submodule=submodule,
            tokenized_inputs_bL=batched_tokens,
            dim_indices=dim_indices,
            dictionary=sae,
        )

    return max_tokens_FKL, max_activations_FKL
