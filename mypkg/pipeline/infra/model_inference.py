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
import mypkg.pipeline.infra.probe_training as probe_training


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
        raise ValueError("Currently broken, see gpu_forward_pass")
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
    save_logits_folder: Optional[str] = None,
    orthogonalize_model: bool = False,
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

    ablation_features_dict = None
    handles = None

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

        ablation_features_dict = {
            hook_layer: (encoder_vectors, decoder_vectors, scales, encoder_biases)
        }

    if ablation_vectors is not None:
        assert ablation_features is None, (
            "Cannot use both ablation_features and ablation_vectors"
        )
        assert ablation_type == "projection_ablations", (
            "ablation_vectors is only supported for projection ablation"
        )
        ablation_features_dict = {}
        for layer, acts_F in ablation_vectors.items():
            ablation_features_dict[layer] = (acts_F, None, None, None)

    if orthogonalize_model:
        assert ablation_vectors is not None, "Cannot orthogonalize model without ablation features"
        model = intervention_hooks.orthogonalize_model_weights(model, ablation_vectors)

    for batch_idx, batch in tqdm(
        enumerate(dataloader), desc="Processing prompts", total=len(dataloader)
    ):
        input_ids, attention_mask, labels, idx_batch, resume_prompt_results_batch = (
            batch
        )
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if ablation_features_dict is not None and not orthogonalize_model:
            handles = []

            for layer_idx, (
                encoder_vectors,
                decoder_vectors,
                scales,
                encoder_biases,
            ) in ablation_features_dict.items():
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

                submodule = model_utils.get_submodule(model, layer_idx)
                handle = submodule.register_forward_hook(ablation_hook)
                handles.append(handle)

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

                if save_logits_folder is not None:
                    save_logits_folder = save_logits_folder.replace("/", "_")
                    os.makedirs(save_logits_folder, exist_ok=True)
                    torch.save(logits_BLV, os.path.join(save_logits_folder, f"logits_{batch_idx}.pt"))
                    print(f"Saved logits of shape {logits_BLV.shape} for batch {batch_idx}")

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
            if handles is not None:
                for handle in handles:
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
    overwrite_previous: bool = False,
) -> torch.Tensor:
    assert padding_side in ["left", "right"]
    assert bias_type in ["gender", "race", "political_orientation"]

    trainer_id = model_utils.MODEL_CONFIGS[model_name]["trainer_id"]

    layer_percent = 25
    downsample = 150

    dataset_name = "resumes"
    dataset_name = "anthropic"

    assert dataset_name in ["resumes", "anthropic"]

    if dataset_name == "anthropic":
        batch_size *= 8

    ablation_features_dir = "ablation_features"
    os.makedirs(ablation_features_dir, exist_ok=True)
    filename = f"ablation_features_{bias_type}_{model_name}_{layer_percent}_{dataset_name}_{trainer_id}_{downsample}.pt".replace(
        "/", "_"
    )
    filename = os.path.join(ablation_features_dir, filename)
    if os.path.exists(filename) and not overwrite_previous:
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
            anthropic_dataset=False,
            downsample=downsample,
            inference_mode="gpu_forward_pass",
            anti_bias_statement_file="v2.txt",
            job_description_file="base_description.txt",
            system_prompt_filename="yes_no.txt",
            bias_type=bias_type,
        )

        if dataset_name == "resumes":
            df = dataset_setup.load_raw_dataset()
            if eval_config.downsample:
                df = dataset_setup.balanced_downsample(
                    df,
                    eval_config.downsample,
                    eval_config.random_seed,
                )
            prompts = hiring_bias_prompts.create_all_prompts_hiring_bias(
                df, eval_config
            )
        elif dataset_name == "anthropic":
            df = dataset_setup.load_full_anthropic_dataset()
            prompts = hiring_bias_prompts.create_all_prompts_anthropic(df, eval_config)

        train_texts, train_labels, train_resume_prompt_results = (
            hiring_bias_prompts.process_hiring_bias_resumes_prompts(
                prompts, model_name, bias_type
            )
        )

        # Assert train_labels only contains 0s and 1s
        assert all(label in [0, 1] for label in train_labels), (
            "train_labels contains values other than 0 and 1"
        )

        # Assert number of 0s equals number of 1s
        num_zeros = sum(1 for label in train_labels if label == 0)
        num_ones = sum(1 for label in train_labels if label == 1)
        assert num_zeros == num_ones, (
            f"train_labels is not balanced: {num_zeros} zeros vs {num_ones} ones"
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
def compute_activations_single_layer(
    transformers_model: AutoModelForCausalLM,
    model_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    chosen_layers: list[int],
    submodules: list[torch.nn.Module],
    final_token_only: bool = True,
    verbose: bool = False,
    ignore_bos: bool = True,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # assert len(chosen_layers) == 1, "Only one layer is supported for now."

    layer_acts_BLD = model_utils.collect_activations(
        transformers_model,
        submodules[0],
        model_inputs,
    )

    layer_acts_BLD *= model_inputs["attention_mask"][:, :, None]

    if final_token_only:
        layer_acts_BLD = layer_acts_BLD[:, -1:, :]

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


def get_single_layer_model_activations(
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


@torch.no_grad()
def compute_activations(
    model: AutoModelForCausalLM,
    model_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    chosen_layers: list[int],
    final_token_only: bool = False,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # assert len(chosen_layers) == 1, "Only one layer is supported for now."

    submodules = [model_utils.get_submodule(model, i) for i in chosen_layers]
    activations_BLD, logits_BLV = model_utils.get_activations_per_layer(
        model, submodules, model_inputs, get_final_token_only=False
    )

    acts_LD = {}
    for i, j in enumerate(chosen_layers):
        acts_LD[j] = activations_BLD[submodules[i]]

    all_acts_D = {}

    for j in chosen_layers:
        layer_acts_BLD = acts_LD[j]
        layer_acts_BLD *= model_inputs["attention_mask"][:, :, None]

        if final_token_only:
            raise NotImplementedError("Final token only not implemented for now")
            layer_acts_BLD = layer_acts_BLD[:, -1:, :]

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

        all_acts_D[j] = (diff_acts_D, pos_acts_D, neg_acts_D)

    return all_acts_D


def get_model_activations(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    chosen_layers: list[int],
) -> dict[int, dict[str, torch.Tensor]]:
    all_acts_D = {}
    for j in chosen_layers:
        all_acts_D[j] = {"diff_acts_D": None, "pos_acts_D": None, "neg_acts_D": None}

    for batch in tqdm(dataloader):
        input_ids, attention_mask, labels, idx_batch, resume_prompt_results_batch = (
            batch
        )

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        batch_acts_D = compute_activations(
            model,
            model_inputs,
            labels,
            chosen_layers,
        )

        for j in chosen_layers:
            if all_acts_D[j]["diff_acts_D"] is None:
                all_acts_D[j]["diff_acts_D"] = torch.zeros_like(
                    batch_acts_D[j][0], dtype=torch.float32
                )
                all_acts_D[j]["pos_acts_D"] = torch.zeros_like(
                    batch_acts_D[j][0], dtype=torch.float32
                )
                all_acts_D[j]["neg_acts_D"] = torch.zeros_like(
                    batch_acts_D[j][0], dtype=torch.float32
                )

                assert (
                    all_acts_D[j]["diff_acts_D"].shape
                    == all_acts_D[j]["pos_acts_D"].shape
                )
                assert (
                    all_acts_D[j]["pos_acts_D"].shape
                    == all_acts_D[j]["neg_acts_D"].shape
                )

            all_acts_D[j]["diff_acts_D"] += batch_acts_D[j][0].to(torch.float32)
            all_acts_D[j]["pos_acts_D"] += batch_acts_D[j][1].to(torch.float32)
            all_acts_D[j]["neg_acts_D"] += batch_acts_D[j][2].to(torch.float32)

    for j in chosen_layers:
        all_acts_D[j]["diff_acts_D"] /= len(dataloader)
        all_acts_D[j]["pos_acts_D"] /= int(len(dataloader) * 0.5)
        all_acts_D[j]["neg_acts_D"] /= int(len(dataloader) * 0.5)

    return all_acts_D


@torch.no_grad()
def compute_mean_activations_per_prompt(
    model: AutoModelForCausalLM,
    model_inputs: dict[str, torch.Tensor],
    race_labels: torch.Tensor,
    gender_labels: torch.Tensor,
    political_orientation_labels: Optional[torch.Tensor],
    chosen_layers: list[int],
    final_token_only: bool = False,
) -> dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    # assert len(chosen_layers) == 1, "Only one layer is supported for now."

    submodules = [model_utils.get_submodule(model, i) for i in chosen_layers]
    activations_BLD, logits_BLV = model_utils.get_activations_per_layer(
        model, submodules, model_inputs, get_final_token_only=False
    )

    acts_LD = {}
    for i, j in enumerate(chosen_layers):
        acts_LD[j] = activations_BLD[submodules[i]]

    all_acts_by_bias_type = {}

    # Process each bias type
    bias_types_and_labels = [
        ("race", race_labels),
        ("gender", gender_labels),
    ]

    if political_orientation_labels is not None:
        bias_types_and_labels.append(
            ("political_orientation", political_orientation_labels)
        )

    for j in chosen_layers:
        layer_acts_BLD = acts_LD[j]
        layer_acts_BLD *= model_inputs["attention_mask"][:, :, None]

        if final_token_only:
            layer_acts_BLD = layer_acts_BLD[:, -1:, :]
            raise NotImplementedError("Final token only not implemented for now")

        layer_bias_acts = {}

        for bias_type, labels in bias_types_and_labels:
            pos_mask_B = labels == 1
            neg_mask_B = labels == 0

            pos_acts_BLD = layer_acts_BLD[pos_mask_B]
            neg_acts_BLD = layer_acts_BLD[neg_mask_B]

            pos_acts_BD = einops.reduce(
                pos_acts_BLD.to(dtype=torch.float32), "b l d -> b d", "mean"
            )
            neg_acts_BD = einops.reduce(
                neg_acts_BLD.to(dtype=torch.float32), "b l d -> b d", "mean"
            )

            if pos_mask_B.sum().item() == 0:
                assert neg_mask_B.sum().item() > 0, (
                    f"No positive or negative examples for {bias_type}"
                )
                pos_acts_BD = None
            elif neg_mask_B.sum().item() == 0:
                assert pos_mask_B.sum().item() > 0, (
                    f"No positive or negative examples for {bias_type}"
                )
                neg_acts_BD = None

            layer_bias_acts[bias_type] = (pos_acts_BD, neg_acts_BD)

        all_acts_by_bias_type[j] = layer_bias_acts

    return all_acts_by_bias_type


@torch.no_grad()
def get_pos_neg_activations(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    chosen_layers: list[int],
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, torch.Tensor]]]:
    pos_acts_BD = {}
    neg_acts_BD = {}

    # Initialize nested structure for each layer and bias type
    for j in chosen_layers:
        pos_acts_BD[j] = {"race": [], "gender": [], "political_orientation": []}
        neg_acts_BD[j] = {"race": [], "gender": [], "political_orientation": []}

    has_political_orientation = False

    first_batch = next(iter(dataloader))
    (
        input_ids,
        attention_mask,
        unused_labels,
        idx_batch,
        resume_prompt_results_batch,
    ) = first_batch

    for resume_prompt_result in resume_prompt_results_batch:
        if resume_prompt_result.political_orientation_added:
            has_political_orientation = True
            break
    bias_types = ["race", "gender"]
    if has_political_orientation:
        bias_types.append("political_orientation")
        raise ValueError("Please review this code")

    for batch in tqdm(dataloader):
        (
            input_ids,
            attention_mask,
            unused_labels,
            idx_batch,
            resume_prompt_results_batch,
        ) = batch

        race_labels = []
        gender_labels = []
        political_orientation_labels = []

        for resume_prompt_result in resume_prompt_results_batch:
            resume_prompt_result: hiring_bias_prompts.ResumePromptResult
            assert resume_prompt_result.race.lower() in ["white", "black"]
            assert resume_prompt_result.gender.lower() in ["male", "female"]

            # Convert boolean to 0/1
            race_labels.append(1 if resume_prompt_result.race.lower() == "black" else 0)
            gender_labels.append(
                1 if resume_prompt_result.gender.lower() == "female" else 0
            )

            if resume_prompt_result.political_orientation_added:
                assert has_political_orientation, "Political orientation not added"
                assert resume_prompt_result.politics.lower() in [
                    "democrat",
                    "republican",
                ]
                political_orientation_labels.append(
                    1 if resume_prompt_result.politics.lower() == "democrat" else 0
                )

        # Convert to tensors
        race_labels = torch.tensor(race_labels, device=input_ids.device)
        gender_labels = torch.tensor(gender_labels, device=input_ids.device)
        political_orientation_labels = (
            torch.tensor(political_orientation_labels, device=input_ids.device)
            if political_orientation_labels
            else None
        )

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        batch_acts_by_bias = compute_mean_activations_per_prompt(
            model,
            model_inputs,
            race_labels,
            gender_labels,
            political_orientation_labels,
            chosen_layers,
        )

        for j in chosen_layers:
            for bias_type in bias_types:
                batch_pos_acts_BD = batch_acts_by_bias[j][bias_type][0]
                batch_neg_acts_BD = batch_acts_by_bias[j][bias_type][1]

                if batch_pos_acts_BD is not None:
                    pos_acts_BD[j][bias_type].append(
                        batch_pos_acts_BD.to(torch.float32)
                    )
                if batch_neg_acts_BD is not None:
                    neg_acts_BD[j][bias_type].append(
                        batch_neg_acts_BD.to(torch.float32)
                    )

    for j in chosen_layers:
        for bias_type in bias_types:
            pos_acts_BD[j][bias_type] = torch.cat(pos_acts_BD[j][bias_type], dim=0)
            neg_acts_BD[j][bias_type] = torch.cat(neg_acts_BD[j][bias_type], dim=0)

    # Remove political_orientation if it was never used
    if not has_political_orientation:
        for j in chosen_layers:
            if "political_orientation" in pos_acts_BD[j]:
                del pos_acts_BD[j]["political_orientation"]
            if "political_orientation" in neg_acts_BD[j]:
                del neg_acts_BD[j]["political_orientation"]

    return pos_acts_BD, neg_acts_BD


def get_probes(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    chosen_layers: list[int],
    lr: float,
    weight_decay: float,
    early_stopping_patience: int,
    max_iter: int,
    probe_batch_size: int,
    testing: bool = False,
) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
    pos_acts_BD, neg_acts_BD = get_pos_neg_activations(model, dataloader, chosen_layers)

    probe_dirs = {}  # layer → bias_type → unit vector (D,)
    probe_accs = {}  # layer → bias_type → test accuracy

    for layer in pos_acts_BD:
        probe_dirs[layer] = {}
        probe_accs[layer] = {}

        # Get bias types available for this layer
        bias_types = list(pos_acts_BD[layer].keys())

        for bias_type in bias_types:
            if bias_type not in neg_acts_BD[layer]:
                raise ValueError(
                    f"Bias type {bias_type} not found in negative activations"
                )

            pos_acts = pos_acts_BD[layer][bias_type]
            neg_acts = neg_acts_BD[layer][bias_type]

            assert len(pos_acts) == len(neg_acts), (
                "Pos and neg acts must have the same number of examples"
            )

            X = torch.cat([pos_acts, neg_acts], dim=0)
            y = torch.cat(
                [
                    torch.ones(len(pos_acts), dtype=torch.long),
                    torch.zeros(len(neg_acts), dtype=torch.long),
                ],
                dim=0,
            )

            # 3. 80/20 split (torch only) --------------------------------------------
            N = X.size(0)
            if testing:
                # This is for determinism on the end to end test
                torch.manual_seed(42)
            idx = torch.randperm(N)
            # split = int(0.8 * N)
            # train_idx, test_idx = idx[:split], idx[split:]
            # X_train, y_train = X[train_idx], y[train_idx]
            # X_test, y_test = X[test_idx], y[test_idx]
            X_train, y_train, X_test, y_test = X, y, X, y

            # 4. train probe ------------------------------------------------------
            gpu_probe, gpu_acc = probe_training.train_probe_gpu(
                X_train.cuda(),
                y_train.cuda(),
                X_test.cuda(),
                y_test.cuda(),
                dim=X.size(1),
                batch_size=probe_batch_size,
                epochs=max_iter,
                lr=lr,
                weight_decay=weight_decay,
                early_stopping_patience=early_stopping_patience,
                verbose=False,
            )
            w_gpu = gpu_probe.net.weight.data.squeeze()
            dir_gpu = w_gpu / w_gpu.norm()

            probe_dirs[layer][bias_type] = {"diff_acts_D": dir_gpu}
            probe_accs[layer][bias_type] = gpu_acc
            print(f"layer {layer:2d} {bias_type:20s} acc {gpu_acc:.3f}")

    return probe_dirs


def get_ablation_vectors(
    model_name: str,
    bias_type: str,  # This parameter might be ignored now since we train all bias types
    eval_config: EvalConfig,
    batch_size: int = 64,
    padding_side: str = "left",
    model: Optional[AutoModelForCausalLM] = None,
) -> dict[int, list[torch.Tensor]]:
    assert padding_side in ["left", "right"]
    # Remove the bias_type assertion since we're training all types

    assert eval_config.probe_training_dataset_name in ["resumes", "anthropic"]

    if eval_config.probe_training_dataset_name == "anthropic" and not eval_config.anthropic_dataset:
        batch_size *= 8

    os.makedirs(eval_config.probe_vectors_dir, exist_ok=True)
    # Update filename to not include specific bias type
    filename = f"ablation_features_all_biases_{model_name}_{eval_config.probe_training_dataset_name}_{eval_config.probe_training_downsample}.pt".replace(
        "/", "_"
    )
    filename = os.path.join(eval_config.probe_vectors_dir, filename)

    if os.path.exists(filename) and not eval_config.probe_training_overwrite_previous:
        all_acts_D = torch.load(filename)
    else:
        print("Computing ablation features for all bias types")
        dtype = torch.bfloat16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                # attn_implementation="flash_attention_2",  # Currently having install issues with flash attention 2
            )

        temp_eval_config = EvalConfig(
            model_name=model_name,
            anthropic_dataset=False,
            downsample=eval_config.probe_training_downsample,
            inference_mode="gpu_forward_pass",
            anti_bias_statement_file=eval_config.probe_training_anti_bias_statement_file,
            job_description_file=eval_config.probe_training_job_description_file,
            system_prompt_filename=eval_config.system_prompt_filename,
            bias_type=bias_type,  # This might not be needed anymore
        )

        if eval_config.probe_training_dataset_name == "resumes":
            df = dataset_setup.load_raw_dataset()
            if eval_config.probe_training_downsample:
                df = dataset_setup.balanced_downsample(
                    df,
                    eval_config.probe_training_downsample,
                    eval_config.random_seed,
                )
            prompts = hiring_bias_prompts.create_all_prompts_hiring_bias(
                df, temp_eval_config
            )
        elif eval_config.probe_training_dataset_name == "anthropic":
            df = dataset_setup.load_full_anthropic_dataset(
                downsample_questions=eval_config.probe_training_downsample
            )
            prompts = hiring_bias_prompts.create_all_prompts_anthropic(
                df, temp_eval_config
            )

        train_texts, unused_labels, train_resume_prompt_results = (
            hiring_bias_prompts.process_hiring_bias_resumes_prompts(
                prompts,
                model_name,
                "race",  # Dummy bias type since we extract all
            )
        )

        dataloader = data_utils.create_simple_dataloader(
            train_texts,
            [0] * len(train_texts),
            train_resume_prompt_results,
            model_name,
            device,
            batch_size=batch_size,
            max_length=eval_config.max_length,
        )

        num_layers = model_utils.get_num_layers(model)
        all_layers = list(range(num_layers))

        # We always train probes on all layers
        all_acts_D = get_probes(
            model,
            dataloader,
            all_layers,
            lr=eval_config.probe_training_lr,
            weight_decay=eval_config.probe_training_weight_decay,
            early_stopping_patience=eval_config.probe_training_early_stopping_patience,
            max_iter=eval_config.probe_training_max_iter,
            probe_batch_size=eval_config.probe_training_batch_size,
            testing=eval_config.test_mode,
        )

        torch.save(all_acts_D, filename)

    # Filter to only the layers we want to use for ablation
    num_layers = max(list(all_acts_D.keys())) + 1
    begin_layer = int(num_layers * eval_config.probe_training_begin_layer_percent / 100)
    chosen_layers = list(range(begin_layer, num_layers))
    all_acts_D = {
        layer: all_acts_D[layer] for layer in all_acts_D if layer in chosen_layers
    }

    # Convert to the expected format: {layer: {bias_type: [vector]}}
    intervention_vectors = {}

    if bias_type == "all":
        first_layer = list(all_acts_D.keys())[0]
        bias_types = list(all_acts_D[first_layer].keys())
    else:
        bias_types = [bias_type]

    for layer in all_acts_D:
        intervention_vectors[layer] = []
        for bias_type in bias_types:
            intervention_vectors[layer].append(
                all_acts_D[layer][bias_type]["diff_acts_D"]
            )

    return intervention_vectors
