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

import mypkg.pipeline.infra.hiring_bias_prompts as hiring_bias_prompts
import mypkg.whitebox_infra.model_utils as model_utils
import mypkg.whitebox_infra.data_utils as data_utils
from mypkg.whitebox_infra.dictionaries import base_sae
import mypkg.whitebox_infra.intervention_hooks as intervention_hooks


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
    padding_side: str = "left",
    max_length: int = 8192,
    model: Optional[AutoModelForCausalLM] = None,
    ablation_type: str = "clamping",
    scale: Optional[float] = None,
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

    for batch in tqdm(dataloader, desc="Processing prompts"):
        input_ids, attention_mask, labels, idx_batch, resume_prompt_results_batch = (
            batch
        )
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if ablation_features is not None:
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
            logits = model(**model_inputs).logits
            if padding_side == "right":
                seq_lengths_B = model_inputs["attention_mask"].sum(dim=1) - 1
                answer_logits_BV = logits[
                    torch.arange(logits.shape[0]),
                    seq_lengths_B,
                    :,
                ]
            elif padding_side == "left":
                answer_logits_BV = logits[
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
