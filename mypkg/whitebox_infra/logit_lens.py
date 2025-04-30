from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import asyncio
from tqdm import tqdm
import json
from datetime import datetime
from typing import Optional, Callable, Literal
import torch
from torch import Tensor
import einops
import torch.nn.functional as F
from dataclasses import asdict

import mypkg.pipeline.infra.hiring_bias_prompts as hiring_bias_prompts
import mypkg.whitebox_infra.model_utils as model_utils
import mypkg.whitebox_infra.data_utils as data_utils


def run_final_block(
    acts_BLD: torch.Tensor,
    model: AutoModelForCausalLM,
) -> torch.Tensor:
    """
    Feeds `acts_BLD` through the last Gemma2DecoderLayer (attention+MLP)
    and returns the updated hidden states.
    """
    B, L, _ = acts_BLD.shape
    device = acts_BLD.device

    pos_ids_BL = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    position_embeddings = model.model.rotary_emb(acts_BLD, pos_ids_BL)

    hidden_BLD = model.model.layers[-1](
        hidden_states=acts_BLD,
        position_embeddings=position_embeddings,
    )
    return hidden_BLD[0]


def apply_logit_lens(
    acts_BLD: torch.Tensor,
    model: AutoModelForCausalLM,
    final_token_only: bool = True,
    add_final_layer: bool = True,
) -> torch.Tensor:
    if final_token_only:
        acts_BLD = acts_BLD[:, -1:, :]

    if add_final_layer:
        acts_BLD += run_final_block(acts_BLD, model)

    logits_BLV = model.lm_head(model.model.norm(acts_BLD))
    return logits_BLV


def kl_between_logits(
    logits_p_BLV: torch.Tensor,  # “teacher” / reference
    logits_q_BLV: torch.Tensor,  # “student” / lens
    reduction: Literal["batchmean", "mean", "none"] = "none",
) -> torch.Tensor:
    """
    KL‖(p || q) for two logit tensors of shape [B, L, V].

    * `p` is obtained with softmax on `logits_p_BLV`
    * `q` is obtained with softmax on `logits_q_BLV`
    * Uses `torch.kl_div(log_q, p)` so we pass log-probabilities for q
    """

    B, L, V = logits_p_BLV.shape

    # Convert logits → probabilities / log-probabilities
    p_BLV = torch.softmax(logits_p_BLV, dim=-1)
    log_q_BLV = torch.log_softmax(logits_q_BLV, dim=-1)

    # KL divergence per token; `reduction` handles batching
    kl = F.kl_div(log_q_BLV, p_BLV, reduction=reduction).sum(dim=-1)

    return kl  # scalar if reduction ≠ "none", else [B, L]


def get_yes_no_statistics(
    logits_BLV: torch.Tensor,
    yes_ids_t: torch.Tensor,
    no_ids_t: torch.Tensor,
    tokenizer: AutoTokenizer,
    k: int = 10,
):
    logits_BV = logits_BLV[:, -1, :]
    probs_BV = torch.softmax(logits_BV, dim=-1)
    yes_logits_BV = logits_BV.index_select(1, yes_ids_t)
    no_logits_BV = logits_BV.index_select(1, no_ids_t)

    yes_probs_BV = probs_BV.index_select(1, yes_ids_t)
    no_probs_BV = probs_BV.index_select(1, no_ids_t)

    yes_logits_B = yes_logits_BV.sum(dim=1)
    no_logits_B = no_logits_BV.sum(dim=1)

    yes_probs_B = yes_probs_BV.sum(dim=1)
    no_probs_B = no_probs_BV.sum(dim=1)

    results = []

    for i in range(logits_BV.shape[0]):
        top_k_vals, top_k_idxs = torch.topk(probs_BV[i], k=k, dim=-1)
        top_tokens = tokenizer.convert_ids_to_tokens(top_k_idxs.tolist())

        results.append(
            {
                "yes_logits": yes_logits_B[i].item(),
                "no_logits": no_logits_B[i].item(),
                "yes_probs": yes_probs_B[i].item(),
                "no_probs": no_probs_B[i].item(),
                "top_k_tokens": top_tokens,
                "top_k_vals": top_k_vals.tolist(),
                "top_k_idxs": top_k_idxs.tolist(),
            }
        )

    return results


@torch.inference_mode()
def run_logit_lens(
    prompt_dicts: list[hiring_bias_prompts.ResumePromptResult],
    model_name: str,
    add_final_layer: bool = False,
    batch_size: int = 64,
    padding_side: str = "left",
    max_length: int = 8192,
    model: Optional[AutoModelForCausalLM] = None,
) -> dict:
    assert padding_side in ["left", "right"]

    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            # attn_implementation="flash_attention_2",  # Currently having install issues with flash attention 2
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    original_prompts = [p.prompt for p in prompt_dicts]

    formatted_prompts = model_utils.add_chat_template(original_prompts, model_name)
    dataloader = data_utils.create_simple_dataloader(
        formatted_prompts,
        [0] * len(formatted_prompts),
        prompt_dicts,
        model_name,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )

    num_layers = len(list(model.model.layers))
    print(f"Number of layers: {num_layers} for {model_name}")

    all_submodules = [model_utils.get_submodule(model, i) for i in range(num_layers)]

    get_final_token_only = True

    yes_ids_t, no_ids_t = model_utils.get_yes_no_ids(tokenizer, device)

    per_layer_results = {}

    for i, layer in enumerate(all_submodules):
        per_layer_results[i] = []
    per_layer_results["output"] = []

    for batch in tqdm(dataloader, desc="Processing prompts"):
        input_ids, attention_mask, labels, idx_batch, resume_prompt_results_batch = (
            batch
        )
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        activations_BLD, logits_BLV = model_utils.get_activations_per_layer(
            model,
            all_submodules,
            model_inputs,
            get_final_token_only=get_final_token_only,
        )

        output_stats = get_yes_no_statistics(logits_BLV, yes_ids_t, no_ids_t, tokenizer)
        for i in range(len(output_stats)):
            output_stats[i]["label"] = labels[i].item()
            output_stats[i]["kl"] = 0.0
            output_stats[i]["resume_prompt_result"] = asdict(
                resume_prompt_results_batch[i]
            )

        per_layer_results["output"].extend(output_stats)

        for i, layer in enumerate(all_submodules):
            logit_lens_BLV = apply_logit_lens(
                activations_BLD[layer],
                model,
                final_token_only=get_final_token_only,
                add_final_layer=add_final_layer,
            )
            kl_BL = kl_between_logits(logits_BLV, logit_lens_BLV)
            kl_B = kl_BL.mean(dim=1)
            layer_stats = get_yes_no_statistics(
                logit_lens_BLV, yes_ids_t, no_ids_t, tokenizer
            )
            for j in range(len(layer_stats)):
                layer_stats[j]["kl"] = kl_B[j].item()
            per_layer_results[i].extend(layer_stats)

    return per_layer_results
