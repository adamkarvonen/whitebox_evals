from transformers import AutoModelForCausalLM
import torch
from typing import Optional

import mypkg.whitebox_infra.dictionaries.base_sae as base_sae
import mypkg.whitebox_infra.dictionaries.batch_topk_sae as batch_topk_sae
import mypkg.whitebox_infra.dictionaries.jumprelu_sae as jumprelu_sae

# Model configuration mapping
MODEL_CONFIGS = {
    "google/gemma-2-9b-it": {
        "total_layers": 40,  # Adding for reference
        "layer_mappings": {
            25: {
                "layer": 9,
                "width_info": "width_16k/average_l0_88",
            },
            50: {
                "layer": 20,
                "width_info": "width_16k/average_l0_91",
            },
            75: {
                "layer": 31,
                "width_info": "width_16k/average_l0_142",
            },
        },
        "batch_size": 4,
    },
    "google/gemma-2-27b-it": {
        "total_layers": 44,  # Adding for reference
        "layer_mappings": {
            25: {
                "layer": 10,
                "width_info": "width_131k/average_l0_106",
            },
            50: {
                "layer": 22,
                "width_info": "width_131k/average_l0_82",
            },
            75: {
                "layer": 34,
                "width_info": "width_131k/average_l0_155",
            },
        },
        "batch_size": 1,
    },
    "mistralai/Ministral-8B-Instruct-2410": {
        "total_layers": 36,
        "layer_mappings": {
            25: {"layer": 9},
            50: {"layer": 18},
            75: {"layer": 27},
        },
        "batch_size": 8,
    },
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "total_layers": 40,
        "layer_mappings": {
            25: {"layer": 10},
            50: {"layer": 20},
            75: {"layer": 30},
        },
        "batch_size": 6,
    },
}


def get_layer_info(model_name: str, layer_percent: int) -> tuple[int, Optional[str]]:
    """Get layer number and width info (if applicable) for a given model and percentage."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not supported")

    layer_mappings = MODEL_CONFIGS[model_name]["layer_mappings"]
    if layer_percent not in layer_mappings:
        raise ValueError(f"Layer percent must be 25, 50, or 75, got {layer_percent}")

    mapping = layer_mappings[layer_percent]
    return mapping["layer"], mapping.get("width_info")


def load_gemma_2_sae(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer_percent: int,
):
    if model_name == "google/gemma-2-9b-it":
        repo_id = "google/gemma-scope-9b-it-res"
    elif model_name == "google/gemma-2-27b-it":
        repo_id = "google/gemma-scope-27b-pt-res"
    else:
        raise ValueError(f"Model {model_name} not supported")

    layer, width_info = get_layer_info(model_name, layer_percent)
    filename = f"layer_{layer}/{width_info}/params.npz"

    sae = jumprelu_sae.load_gemma_scope_jumprelu_sae(
        repo_id=repo_id,
        filename=filename,
        layer=layer,
        model_name=model_name,
        device=device,
        dtype=dtype,
        local_dir="downloaded_saes",
    )
    return sae


def load_mistral_sae(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer_percent: int,
    trainer_id: int = 1,
):
    layer, _ = get_layer_info(model_name, layer_percent)
    if model_name == "mistralai/Ministral-8B-Instruct-2410":
        sae_repo = "adamkarvonen/ministral_saes"
        sae_path = f"mistralai_Ministral-8B-Instruct-2410_batch_top_k/resid_post_layer_{layer}/trainer_{trainer_id}/ae.pt"
    elif model_name == "mistralai/Mistral-Small-24B-Instruct-2501":
        sae_repo = "adamkarvonen/mistral_24b_saes"
        sae_path = f"mistral_24b_mistralai_Mistral-Small-24B-Instruct-2501_batch_top_k/resid_post_layer_{layer}/trainer_{trainer_id}/ae.pt"
    else:
        raise ValueError(f"Model {model_name} not supported")

    sae = batch_topk_sae.load_dictionary_learning_batch_topk_sae(
        repo_id=sae_repo,
        filename=sae_path,
        model_name=model_name,
        device=device,
        dtype=dtype,
        layer=layer,
        local_dir="downloaded_saes",
    )
    return sae


def load_model_sae(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer_percent: int,
    trainer_id: Optional[int] = None,
) -> base_sae.BaseSAE:
    if layer_percent not in (25, 50, 75):
        raise ValueError(f"Layer percent must be 25, 50, or 75, got {layer_percent}")

    if model_name == "google/gemma-2-9b-it" or model_name == "google/gemma-2-27b-it":
        return load_gemma_2_sae(model_name, device, dtype, layer_percent)
    elif (
        model_name == "mistralai/Ministral-8B-Instruct-2410"
        or model_name == "mistralai/Mistral-Small-24B-Instruct-2501"
    ):
        return load_mistral_sae(model_name, device, dtype, layer_percent, trainer_id)
    else:
        raise ValueError(f"Model {model_name} not supported")


def add_mistral_v3_chat_template(prompts: list[str]) -> list[str]:
    return [f"[INST]{prompt}[/INST]" for prompt in prompts]


def add_gemma_chat_template(prompts: list[str]) -> list[str]:
    return [
        f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        for prompt in prompts
    ]


def add_chat_template(prompts: list[str], model_name: str) -> list[str]:
    if (
        model_name == "mistralai/Ministral-8B-Instruct-2410"
        or model_name == "mistralai/Mistral-Small-24B-Instruct-2501"
    ):
        return add_mistral_v3_chat_template(prompts)
    elif model_name == "google/gemma-2-9b-it" or model_name == "google/gemma-2-27b-it":
        return add_gemma_chat_template(prompts)
    else:
        raise ValueError(f"Please implement a chat template for {model_name}")


def get_submodule(model: AutoModelForCausalLM, layer: int):
    """Gets the residual stream submodule"""
    model_name = model.config._name_or_path

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    elif "gemma" in model_name or "mistral" in model_name:
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")


class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""

    pass


@torch.no_grad()
def collect_activations(
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    inputs_BL: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Registers a forward hook on the submodule to capture the residual (or hidden)
    activations. We then raise an EarlyStopException to skip unneeded computations.
    """
    activations_BLD = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal activations_BLD
        # For many models, the submodule outputs are a tuple or a single tensor:
        # If "outputs" is a tuple, pick the relevant item:
        #   e.g. if your layer returns (hidden, something_else), you'd do outputs[0]
        # Otherwise just do outputs
        if isinstance(outputs, tuple):
            activations_BLD = outputs[0]
        else:
            activations_BLD = outputs

        raise EarlyStopException("Early stopping after capturing activations")

    handle = submodule.register_forward_hook(gather_target_act_hook)

    try:
        _ = model(**inputs_BL)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        handle.remove()

    return activations_BLD
