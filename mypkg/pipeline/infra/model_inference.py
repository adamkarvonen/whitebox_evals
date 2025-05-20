# mypkg/pipeline/infra/model_inference.py

import os
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
import asyncio
from tqdm import tqdm
import json
from datetime import datetime
from typing import Optional, Callable
import torch
from jaxtyping import Float
from torch import Tensor
import einops
import argparse
from torch.utils.data import DataLoader

import mypkg.pipeline.infra.hiring_bias_prompts as hiring_bias_prompts
import mypkg.whitebox_infra.model_utils as model_utils
import mypkg.whitebox_infra.data_utils as data_utils
from mypkg.whitebox_infra.dictionaries import base_sae
import mypkg.whitebox_infra.intervention_hooks as intervention_hooks
from mypkg.eval_config import EvalConfig
import mypkg.pipeline.setup.dataset as dataset_setup


async def openrouter_request(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    max_completion_tokens: Optional[int] = None,
    timeout_seconds: float = 120.0,
) -> tuple[str, Optional[dict]]:
    try:
        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_completion_tokens,
                temperature=0.0,
            ),
            timeout=timeout_seconds,
        )
        message = completion.choices[0].message.content
        completion = completion.model_dump()
    except asyncio.TimeoutError:
        message = f"Error: request timed out after {timeout_seconds} seconds"
        completion = None
    except Exception as e:
        message = f"Error: {e}"
        completion = None

    return message, completion


async def run_all_prompts(
    client: openai.OpenAI,
    api_llm: str,
    prompt_dicts: list[hiring_bias_prompts.ResumePromptResult],
    max_completion_tokens: Optional[int] = None,
    timeout_seconds: float = 120.0,
) -> list[hiring_bias_prompts.ResumePromptResult]:
    tasks = [
        openrouter_request(
            client, api_llm, prompt_dict.prompt, max_completion_tokens, timeout_seconds
        )
        for prompt_dict in prompt_dicts
    ]

    # Track completed tasks in a background coroutine
    done_count = 0

    async def wrapped_task(t):
        nonlocal done_count
        res = await t
        done_count += 1
        pbar.update(1)
        return res

    with tqdm(total=len(tasks), desc="Processing prompts") as pbar:
        wrapped_tasks = [wrapped_task(task) for task in tasks]
        results = await asyncio.gather(*wrapped_tasks)

    for i, result in enumerate(results):
        prompt_dicts[i].response = result[0]
        prompt_dicts[i].chat_completion = result[1]

    return prompt_dicts


async def run_model_inference_openrouter(
    prompt_dicts: list[hiring_bias_prompts.ResumePromptResult],
    api_llm: str,
    max_completion_tokens: Optional[int] = None,
    timeout_seconds: float = 120.0,
) -> list[hiring_bias_prompts.ResumePromptResult]:
    """
    Sends prompts to OpenRouter API and returns the responses.
    """
    with open("openrouter_api_key.txt", "r") as f:
        API_KEY = f.read().strip()

    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
    )
    results = await run_all_prompts(
        client, api_llm, prompt_dicts, max_completion_tokens, timeout_seconds
    )

    await client.close()

    return results


@torch.inference_mode()
def run_inference_vllm(
    prompt_dicts: list[hiring_bias_prompts.ResumePromptResult],
    model_name: str,
    max_new_tokens: int = 200,
    max_length: int = 8192,
    model=None,
) -> list[hiring_bias_prompts.ResumePromptResult]:
    import vllm

    if model is None:
        model = vllm.LLM(model=model_name, dtype="bfloat16")
    original_prompts = [p.prompt for p in prompt_dicts]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        print("No pad token found, setting eos token as pad token")
        tokenizer.pad_token = tokenizer.eos_token

    # Format prompts with chat template if needed
    formatted_prompts = model_utils.add_chat_template(original_prompts, model_name)
    tokenized_inputs = tokenizer(
        formatted_prompts,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    prompt_token_ids = [input_ids for input_ids in tokenized_inputs["input_ids"]]

    # Create sampling parameters
    sampling_params = vllm.SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,  # Equivalent to do_sample=False
    )

    outputs = model.generate(
        prompt_token_ids=prompt_token_ids,  # Pass raw strings directly
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    for i, output in enumerate(outputs):
        # Get the generated token IDs and decode
        generated_ids = output.outputs[0].token_ids
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        prompt_dicts[i].response = response_text

    # This is to use a list of strings
    # outputs = model.generate(
    #     prompts=formatted_prompts,  # Pass raw strings directly
    #     sampling_params=sampling_params,
    #     use_tqdm=True,
    # )

    # Process outputs
    # for i, output in enumerate(outputs):
    #     # Get the generated text directly (vLLM decodes it for us)
    #     response_text = output.outputs[0].text
    #     prompt_dicts[i].response = response_text

    return prompt_dicts


@torch.inference_mode()
def run_inference_transformers(
    prompt_dicts: list[hiring_bias_prompts.ResumePromptResult],
    model_name: str,
    batch_size: int = 64,
    ablation_features: Optional[torch.Tensor] = None,
    max_new_tokens: int = 200,
    max_length: int = 8192,
) -> list[hiring_bias_prompts.ResumePromptResult]:
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        # attn_implementation="flash_attention_2",  # FlashAttention2 doesn't support right padding with mistral
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        print("No pad token found, setting eos token as pad token")
        tokenizer.pad_token = tokenizer.eos_token

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

    if ablation_features is not None:
        sae = model_utils.load_model_sae(model_name, device, dtype, 25, trainer_id=1)
        encoder_vectors, decoder_vectors, encoder_biases = (
            intervention_hooks.get_sae_vectors(ablation_features, sae)
        )
        scales = [0.0] * len(encoder_vectors)
        ablation_hook = intervention_hooks.get_conditional_clamping_hook(
            encoder_vectors, decoder_vectors, scales, encoder_biases
        )
        submodule = model_utils.get_submodule(model, sae.hook_layer)
        handle = submodule.register_forward_hook(ablation_hook)

    try:
        for batch in tqdm(dataloader, desc="Processing prompts"):
            (
                input_ids,
                attention_mask,
                labels,
                idx_batch,
                resume_prompt_results_batch,
            ) = batch
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            response = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            response = response[:, input_ids.shape[1] :]
            response = tokenizer.batch_decode(response, skip_special_tokens=True)

            for i, idx in enumerate(idx_batch):
                prompt_dicts[idx].response = response[i]
    finally:
        if ablation_features is not None:
            handle.remove()

    return prompt_dicts


@torch.inference_mode()
def run_single_forward_pass_transformers(
    prompt_dicts: list[hiring_bias_prompts.ResumePromptResult],
    model_name: str,
    batch_size: int = 64,
    ablation_features: Optional[torch.Tensor] = None,
    ablation_vectors: Optional[dict[int, list[torch.Tensor]]] = None,
    padding_side: str = "left",
    max_length: int = 8192,
    model: Optional[AutoModelForCausalLM] = None,
    ablation_type: str = "clamping",
    scale: Optional[float] = None,
    collect_activations: bool = False,
) -> list[hiring_bias_prompts.ResumePromptResult]:
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

    if ablation_features is not None:
        trainer_id = model_utils.MODEL_CONFIGS[model_name]["trainer_id"]
        sae = model_utils.load_model_sae(
            model_name, device, dtype, 25, trainer_id=trainer_id
        )
        scales = [scale] * len(ablation_features)
        encoder_vectors, decoder_vectors, encoder_biases = (
            intervention_hooks.get_sae_vectors(ablation_features, sae)
        )
        sae = sae.to("cpu")
        hook_layer = sae.hook_layer
        del sae
        torch.cuda.empty_cache()

    if ablation_vectors is not None:
        assert ablation_features is None, (
            "Cannot use both ablation_features and ablation_vectors"
        )
        assert ablation_type == "projection_ablations", (
            "ablation_vectors is only supported for projection ablation"
        )

        assert len(ablation_vectors.keys()) == 1, "only support one layer for now"

        hook_layer = list(ablation_vectors.keys())[0]
        scales = None
        encoder_vectors = ablation_vectors[hook_layer]
        decoder_vectors = None
        encoder_biases = None

    for batch in tqdm(dataloader, desc="Processing prompts"):
        input_ids, attention_mask, labels, idx_batch, resume_prompt_results_batch = (
            batch
        )
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if ablation_features is not None or ablation_vectors is not None:
            ablation_hook = intervention_hooks.get_ablation_hook(
                ablation_type,
                encoder_vectors,
                decoder_vectors,
                scales,
                encoder_biases,
                resume_prompt_results_batch,
                input_ids,
                tokenizer,
            )

            submodule = model_utils.get_submodule(model, hook_layer)
            handle = submodule.register_forward_hook(ablation_hook)

        try:
            if collect_activations:
                submodules = [
                    model_utils.get_submodule(model, i)
                    for i in range(len(list(model.model.layers)))
                ]
                activations_BLD, logits_BLV = model_utils.get_activations_per_layer(
                    model, submodules, model_inputs, get_final_token_only=True
                )

                for i, idx in enumerate(idx_batch):
                    acts_LD = {}
                    for j in range(len(submodules)):
                        acts_LD[j] = activations_BLD[submodules[j]][i].cpu()

                    prompt_dicts[idx].activations = acts_LD
            else:
                logits_BLV = model(**model_inputs).logits

            if padding_side == "right":
                seq_lengths_B = model_inputs["attention_mask"].sum(dim=1) - 1
                answer_logits_BV = logits_BLV[
                    torch.arange(logits_BLV.shape[0]),
                    seq_lengths_B,
                    :,
                ]
            elif padding_side == "left":
                answer_logits_BV = logits_BLV[
                    :,
                    -1,
                    :,
                ]
            answer_logits_B = torch.argmax(answer_logits_BV, dim=-1)
            predicted_tokens = tokenizer.batch_decode(answer_logits_B.unsqueeze(-1))
            yes_probs_B, no_probs_B = model_utils.get_yes_no_probs(
                tokenizer, answer_logits_BV
            )

            for i, idx in enumerate(idx_batch):
                prompt_dicts[idx].response = predicted_tokens[i]
                prompt_dicts[idx].yes_probs = yes_probs_B[i].item()
                prompt_dicts[idx].no_probs = no_probs_B[i].item()
        finally:
            if ablation_features is not None:
                handle.remove()

    return prompt_dicts


@torch.no_grad()
def compute_sae_activations(
    transformers_model: AutoModelForCausalLM,
    sae: base_sae.BaseSAE,
    model_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    chosen_layers: list[int],
    submodules: list[torch.nn.Module],
    verbose: bool = False,
    ignore_bos: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(chosen_layers) == 1, "Only one layer is supported for now."

    layer_acts_BLD = model_utils.collect_activations(
        transformers_model,
        submodules[0],
        model_inputs,
    )

    encoded_acts_BLF = sae.encode(layer_acts_BLD)

    encoded_acts_BLF *= model_inputs["attention_mask"][:, :, None]

    # encoded_acts_BLF = encoded_acts_BLF[:, -10:, :]

    pos_mask_B = labels == 1
    neg_mask_B = labels == 0

    pos_acts_BLF = encoded_acts_BLF[pos_mask_B]
    neg_acts_BLF = encoded_acts_BLF[neg_mask_B]

    pos_acts_F = einops.reduce(
        pos_acts_BLF.to(dtype=torch.float32), "b l f -> f", "mean"
    )
    neg_acts_F = einops.reduce(
        neg_acts_BLF.to(dtype=torch.float32), "b l f -> f", "mean"
    )

    if pos_mask_B.sum().item() == 0:
        assert neg_mask_B.sum().item() > 0, "No positive or negative examples"
        pos_acts_F = torch.zeros_like(neg_acts_F)
    if neg_mask_B.sum().item() == 0:
        assert pos_mask_B.sum().item() > 0, "No positive or negative examples"
        neg_acts_F = torch.zeros_like(pos_acts_F)

    diff_acts_F = pos_acts_F - neg_acts_F

    return diff_acts_F, pos_acts_F, neg_acts_F


def get_sae_activations(
    model: AutoModelForCausalLM,
    sae: base_sae.BaseSAE,
    dataloader: DataLoader,
    submodules: list[torch.nn.Module],
    chosen_layers: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff_acts_F = None  # will hold ΣΔf
    pos_acts_F = None  # will hold Σpos
    neg_acts_F = None  # will hold Σneg

    for batch in tqdm(dataloader):
        input_ids, attention_mask, labels, idx_batch, resume_prompt_results_batch = (
            batch
        )

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        batch_diff_acts_F, batch_pos_acts_F, batch_neg_acts_F = compute_sae_activations(
            model,
            sae,
            model_inputs,
            labels,
            chosen_layers,
            submodules,
            verbose=False,
        )

        if diff_acts_F is None:
            diff_acts_F = torch.zeros_like(batch_diff_acts_F, dtype=torch.float32)
            pos_acts_F = torch.zeros_like(batch_pos_acts_F, dtype=torch.float32)
            neg_acts_F = torch.zeros_like(batch_neg_acts_F, dtype=torch.float32)

            assert diff_acts_F.shape == pos_acts_F.shape
            assert pos_acts_F.shape == neg_acts_F.shape

        diff_acts_F += batch_diff_acts_F.to(torch.float32)
        pos_acts_F += batch_pos_acts_F.to(torch.float32)
        neg_acts_F += batch_neg_acts_F.to(torch.float32)

    diff_acts_F /= len(dataloader)
    pos_acts_F /= int(len(dataloader) * 0.5)
    neg_acts_F /= int(len(dataloader) * 0.5)

    return diff_acts_F, pos_acts_F, neg_acts_F


def get_ablation_features(
    model_name: str,
    bias_type: str,
    batch_size: int = 64,
    padding_side: str = "left",
    max_length: int = 8192,
    model: Optional[AutoModelForCausalLM] = None,
) -> torch.Tensor:
    assert padding_side in ["left", "right"]
    assert bias_type in ["gender", "race", "political_orientation"]

    trainer_id = model_utils.MODEL_CONFIGS[model_name]["trainer_id"]

    layer_percent = 25
    downsample = 150

    ablation_features_dir = "ablation_features"
    os.makedirs(ablation_features_dir, exist_ok=True)
    filename = f"ablation_features_{bias_type}_{model_name}_{layer_percent}_{trainer_id}_{downsample}.pt".replace(
        "/", "_"
    )
    filename = os.path.join(ablation_features_dir, filename)
    if os.path.exists(filename):
        diff_acts_F, pos_acts_F, neg_acts_F = torch.load(filename)
    else:
        print("Computing ablation features")
        dtype = torch.bfloat16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                # attn_implementation="flash_attention_2",  # Currently having install issues with flash attention 2
            )
        # tokenizer = AutoTokenizer.from_pretrained(model_name)

        eval_config = EvalConfig(
            model_name=model_name,
            political_orientation=bias_type == "political_orientation",
            pregnancy=False,
            employment_gap=False,
            anthropic_dataset=False,
            downsample=downsample,
            inference_mode="gpu_forward_pass",
            anti_bias_statement_file="v2.txt",
            job_description_file="base_description.txt",
            system_prompt_filename="yes_no.txt",
        )

        args = hiring_bias_prompts.HiringBiasArgs(
            political_orientation=bias_type == "political_orientation",
            employment_gap=bias_type == "employment_gap",
            pregnancy=bias_type == "pregnancy",
            race=bias_type == "race",
            gender=bias_type == "gender",
            misc=bias_type == "misc",
        )

        df = dataset_setup.load_raw_dataset()
        if eval_config.downsample:
            df = dataset_setup.balanced_downsample(
                df,
                eval_config.downsample,
                eval_config.random_seed,
            )

        prompts = hiring_bias_prompts.create_all_prompts_hiring_bias(
            df, args, eval_config
        )

        train_texts, train_labels, train_resume_prompt_results = (
            hiring_bias_prompts.process_hiring_bias_resumes_prompts(
                prompts, model_name, args
            )
        )

        dataloader = data_utils.create_simple_dataloader(
            train_texts,
            train_labels,
            train_resume_prompt_results,
            model_name,
            device,
            batch_size=batch_size,
            max_length=max_length,
        )

        sae = model_utils.load_model_sae(
            model_name, device, dtype, layer_percent, trainer_id=trainer_id
        )

        chosen_layer = model_utils.MODEL_CONFIGS[model_name]["layer_mappings"][
            layer_percent
        ]["layer"]

        submodules = [model_utils.get_submodule(model, chosen_layer)]

        diff_acts_F, pos_acts_F, neg_acts_F = get_sae_activations(
            model, sae, dataloader, submodules, [chosen_layer]
        )

        torch.save((diff_acts_F, pos_acts_F, neg_acts_F), filename)

    acts_top_k_ids = diff_acts_F.abs().topk(100).indices

    # set torch print sci mode false
    # torch.set_printoptions(sci_mode=False)

    ratios = pos_acts_F[acts_top_k_ids] / (neg_acts_F[acts_top_k_ids] + 1e-10)

    mask = (ratios > 2.0) | (ratios < 0.5)

    selected_ids = acts_top_k_ids[mask]

    return selected_ids


@torch.no_grad()
def compute_activations(
    transformers_model: AutoModelForCausalLM,
    model_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    chosen_layers: list[int],
    submodules: list[torch.nn.Module],
    verbose: bool = False,
    ignore_bos: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(chosen_layers) == 1, "Only one layer is supported for now."

    layer_acts_BLD = model_utils.collect_activations(
        transformers_model,
        submodules[0],
        model_inputs,
    )

    layer_acts_BLD *= model_inputs["attention_mask"][:, :, None]

    # encoded_acts_BLF = encoded_acts_BLF[:, -10:, :]

    pos_mask_B = labels == 1
    neg_mask_B = labels == 0

    pos_acts_BLD = layer_acts_BLD[pos_mask_B]
    neg_acts_BLD = layer_acts_BLD[neg_mask_B]

    pos_acts_D = einops.reduce(
        pos_acts_BLD.to(dtype=torch.float32), "b l d -> d", "mean"
    )
    neg_acts_D = einops.reduce(
        neg_acts_BLD.to(dtype=torch.float32), "b l d -> d", "mean"
    )

    if pos_mask_B.sum().item() == 0:
        assert neg_mask_B.sum().item() > 0, "No positive or negative examples"
        pos_acts_D = torch.zeros_like(neg_acts_D)
    if neg_mask_B.sum().item() == 0:
        assert pos_mask_B.sum().item() > 0, "No positive or negative examples"
        neg_acts_D = torch.zeros_like(pos_acts_D)

    diff_acts_D = pos_acts_D - neg_acts_D

    return diff_acts_D, pos_acts_D, neg_acts_D


def get_model_activations(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    submodules: list[torch.nn.Module],
    chosen_layers: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    diff_acts_D = None  # will hold ΣΔf
    pos_acts_D = None  # will hold Σpos
    neg_acts_D = None  # will hold Σneg

    for batch in tqdm(dataloader):
        input_ids, attention_mask, labels, idx_batch, resume_prompt_results_batch = (
            batch
        )

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        batch_diff_acts_D, batch_pos_acts_D, batch_neg_acts_D = compute_activations(
            model,
            model_inputs,
            labels,
            chosen_layers,
            submodules,
            verbose=False,
        )

        if diff_acts_D is None:
            diff_acts_D = torch.zeros_like(batch_diff_acts_D, dtype=torch.float32)
            pos_acts_D = torch.zeros_like(batch_pos_acts_D, dtype=torch.float32)
            neg_acts_D = torch.zeros_like(batch_neg_acts_D, dtype=torch.float32)

            assert diff_acts_D.shape == pos_acts_D.shape
            assert pos_acts_D.shape == neg_acts_D.shape

        diff_acts_D += batch_diff_acts_D.to(torch.float32)
        pos_acts_D += batch_pos_acts_D.to(torch.float32)
        neg_acts_D += batch_neg_acts_D.to(torch.float32)

    diff_acts_D /= len(dataloader)
    pos_acts_D /= int(len(dataloader) * 0.5)
    neg_acts_D /= int(len(dataloader) * 0.5)

    return diff_acts_D, pos_acts_D, neg_acts_D


def get_ablation_vectors(
    model_name: str,
    bias_type: str,
    batch_size: int = 64,
    padding_side: str = "left",
    max_length: int = 8192,
    model: Optional[AutoModelForCausalLM] = None,
) -> dict[int, list[torch.Tensor]]:
    assert padding_side in ["left", "right"]
    assert bias_type in ["gender", "race", "political_orientation"]

    layer_percent = 25
    downsample = 150

    chosen_layer = model_utils.MODEL_CONFIGS[model_name]["layer_mappings"][
        layer_percent
    ]["layer"]

    ablation_features_dir = "ablation_vectors"
    os.makedirs(ablation_features_dir, exist_ok=True)
    filename = f"ablation_features_{bias_type}_{model_name}_{layer_percent}_{downsample}.pt".replace(
        "/", "_"
    )
    filename = os.path.join(ablation_features_dir, filename)
    if os.path.exists(filename):
        diff_acts_F, pos_acts_F, neg_acts_F = torch.load(filename)
    else:
        print("Computing ablation features")
        dtype = torch.bfloat16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                # attn_implementation="flash_attention_2",  # Currently having install issues with flash attention 2
            )
        # tokenizer = AutoTokenizer.from_pretrained(model_name)

        eval_config = EvalConfig(
            model_name=model_name,
            political_orientation=bias_type == "political_orientation",
            pregnancy=False,
            employment_gap=False,
            anthropic_dataset=False,
            downsample=downsample,
            inference_mode="gpu_forward_pass",
            anti_bias_statement_file="v2.txt",
            job_description_file="base_description.txt",
            system_prompt_filename="yes_no.txt",
        )

        args = hiring_bias_prompts.HiringBiasArgs(
            political_orientation=bias_type == "political_orientation",
            employment_gap=bias_type == "employment_gap",
            pregnancy=bias_type == "pregnancy",
            race=bias_type == "race",
            gender=bias_type == "gender",
            misc=bias_type == "misc",
        )

        df = dataset_setup.load_raw_dataset()
        if eval_config.downsample:
            df = dataset_setup.balanced_downsample(
                df,
                eval_config.downsample,
                eval_config.random_seed,
            )

        prompts = hiring_bias_prompts.create_all_prompts_hiring_bias(
            df, args, eval_config
        )

        train_texts, train_labels, train_resume_prompt_results = (
            hiring_bias_prompts.process_hiring_bias_resumes_prompts(
                prompts, model_name, args
            )
        )

        dataloader = data_utils.create_simple_dataloader(
            train_texts,
            train_labels,
            train_resume_prompt_results,
            model_name,
            device,
            batch_size=batch_size,
            max_length=max_length,
        )

        submodules = [model_utils.get_submodule(model, chosen_layer)]

        diff_acts_F, pos_acts_F, neg_acts_F = get_model_activations(
            model, dataloader, submodules, [chosen_layer]
        )

        torch.save((diff_acts_F, pos_acts_F, neg_acts_F), filename)

    return {chosen_layer: [diff_acts_F]}
