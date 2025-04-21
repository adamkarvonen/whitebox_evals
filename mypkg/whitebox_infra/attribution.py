import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional, Callable
import torch
import einops

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from mypkg.whitebox_infra.dictionaries import topk_sae, base_sae
import mypkg.whitebox_infra.model_utils as model_utils


def collect_token_ids(tokenizer, candidates):
    """
    Collect the set of token IDs that appear when tokenizing
    any candidate string in `candidates`.
    """
    token_ids = set()
    for c in candidates:
        # add_special_tokens=False so we only get raw subwords
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) > 1:
            ids = ids[:1]
        assert len(ids) == 1, f"Expected 1 token ID, got {len(ids)} for {c}"
        token_ids.update(ids)

    return list(token_ids)


def get_yes_no_ids(
    tokenizer, yes_candidates: list[str], no_candidates: list[str], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    yes_ids = collect_token_ids(tokenizer, yes_candidates)
    no_ids = collect_token_ids(tokenizer, no_candidates)

    yes_ids_t = torch.tensor(yes_ids).to(device)  # for indexing
    no_ids_t = torch.tensor(no_ids).to(device)

    return yes_ids_t, no_ids_t


def make_yes_no_loss_fn(
    tokenizer,
    device: torch.device,
    yes_candidates: list[str],
    no_candidates: list[str],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    # Collect sets of IDs
    yes_ids_t, no_ids_t = get_yes_no_ids(
        tokenizer, yes_candidates, no_candidates, device
    )

    def yes_no_loss_fn(next_token_logits_BLV: torch.Tensor, labels_B: torch.Tensor):
        next_token_logits_BV = next_token_logits_BLV[:, -1, :]
        # Gather the logits for yes_ids and no_ids
        # .index_select(1, ...) means we select columns by ID
        yes_logits = next_token_logits_BV.index_select(1, yes_ids_t)
        no_logits = next_token_logits_BV.index_select(1, no_ids_t)

        # Sum over the sets
        yes_sum = yes_logits.sum(dim=1)  # shape [batch_size]
        no_sum = no_logits.sum(dim=1)

        # difference per example
        diff = yes_sum - no_sum  # shape [batch_size]
        adjusted_diff = diff

        # Apply the label-dependent logic:
        # For label=0: use diff as is
        # For label=1: flip the sign of diff
        label_factor = 1 - 2 * labels_B.float()  # Convert 0->1, 1->-1
        adjusted_diff = diff * label_factor

        return adjusted_diff.mean()

    return yes_no_loss_fn


def make_max_logprob_loss_fn() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def max_logprob_loss_fn(
        next_token_logits_BLV: torch.Tensor, labels_B: torch.Tensor
    ):
        next_token_logits_BV = next_token_logits_BLV[:, -1, :]
        # convert logits to log‑probs
        log_probs = torch.nn.functional.log_softmax(next_token_logits_BV, dim=-1)
        # highest‑probability token per example
        max_log_probs_B = log_probs.max(dim=1).values  # shape [batch_size]

        # Apply the label-dependent logic:
        # For label=0: use diff as is
        # For label=1: flip the sign of diff
        label_factor = 1 - 2 * labels_B.float()  # Convert 0->1, 1->-1
        max_log_probs_B = max_log_probs_B * label_factor

        return -max_log_probs_B.mean()  # negate so lower = better

    return max_logprob_loss_fn


def make_activation_loss_fn(
    seq_slice: slice = slice(-1, None),  # by default use the last two tokens
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Self‑energy loss with optional label flipping.

    For every example:
        E = ½ * sum_{t in seq_slice} ||h_t||²
    If labels_B[b] == 1 the sign of E is flipped.

    The returned loss is  -mean(E_signed) so that   loss ↓  ⇒
        • smaller energy for label 0
        • larger (negated) energy for label 1

    Args
    ----
    device      : torch.device (kept for API symmetry; unused here)
    seq_slice   : slice object selecting the sequence positions to include.
                  Default `slice(-2, None)` uses the last two tokens.

    Returns
    -------
    activation_loss_fn : Callable(activations_BLD, labels_B) -> scalar loss
    """

    def activation_loss_fn(
        activations_BLD: torch.Tensor,  # [B, L, D]
        labels_B: torch.Tensor,  # [B]  (0 or 1)
    ) -> torch.Tensor:
        # 1. pick the desired sequence positions
        h = activations_BLD[:, seq_slice, :]  # [B, L_sel, D]

        # 2. self‑energy per example
        energy_B = 0.5 * (h**2).sum(dim=(-1, -2))  # sum over D and L_sel  → [B]

        # 3. label‑dependent sign flip  (0 -> +1, 1 -> –1)
        label_factor = 1.0 - 2.0 * labels_B.float().to(energy_B.device)
        signed_energy_B = energy_B * label_factor

        # 4. final loss: negate so "lower = better"
        return -signed_energy_B.mean()

    return activation_loss_fn


def view_outputs(
    tokenizer, all_input_ids: torch.Tensor, all_answer_logits: torch.Tensor
):
    for i in range(all_input_ids.shape[0]):
        input_ids = all_input_ids[i]
        answer_logits = all_answer_logits[i : i + 1, :]

        print()
        print(tokenizer.decode(input_ids))
        # Compute log probabilities (numerically stable)
        log_probs = torch.nn.functional.log_softmax(answer_logits, dim=-1)
        # Exponentiate log probabilities to get actual probabilities
        probs = torch.exp(log_probs)

        # Set the number of top tokens you want to inspect
        topk = 10

        for batch_idx in range(answer_logits.shape[0]):
            # Get top-k log probabilities and corresponding indices for this batch element
            top_percentages, top_indices = torch.topk(probs[batch_idx], topk)
            # Exponentiate to obtain probabilities and convert to percentages
            top_percentages *= 100

            # Convert token IDs to token strings
            top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())

            print(f"Batch element {batch_idx}:")
            for token, token_id, percentage in zip(
                top_tokens, top_indices.tolist(), top_percentages.tolist(), strict=True
            ):
                print(
                    f"Token: {token:15s} | Token ID: {token_id:4d} | Probability: {percentage:6.2f}%"
                )
            print("-" * 50)


def analyze_prediction_rates(
    labels: list[int], predicted_tokens: list[str], verbose: bool = False
) -> dict[int, dict]:
    # Initialize counters
    stats = {
        1: {"yes": 0, "no": 0, "invalid": 0, "total": 0},
        0: {"yes": 0, "no": 0, "invalid": 0, "total": 0},
    }

    # Count occurrences
    for label, pred in zip(labels, predicted_tokens):
        stats[label]["total"] += 1
        pred = pred.lower()
        if pred == "yes":
            stats[label]["yes"] += 1
        elif pred == "no":
            stats[label]["no"] += 1
        else:
            stats[label]["invalid"] += 1

    # Calculate rates
    results = {}
    for label in stats:
        total = stats[label]["total"]
        results[label] = {
            "yes_rate": stats[label]["yes"] / total * 100,
            "no_rate": stats[label]["no"] / total * 100,
            "invalid_rate": stats[label]["invalid"] / total * 100,
            "total_samples": total,
        }

    # Print results in a readable format
    if verbose:
        for label in stats:
            print(f"\nLabel {label}:")
            print(f"Total samples: {results[label]['total_samples']}")
            print(f"Yes rate: {results[label]['yes_rate']:.1f}%")
            print(f"No rate: {results[label]['no_rate']:.1f}%")
            print(f"Invalid rate: {results[label]['invalid_rate']:.1f}%")

    return results


def compute_attributions(
    transformers_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae: base_sae.BaseSAE,
    model_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    chosen_layers: list[int],
    submodules: list[torch.nn.Module],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    use_stop_gradient: bool = False,
    use_activation_loss_fn: bool = False,
    verbose: bool = False,
    ignore_bos: bool = True,
    padding_side: str = "left",
) -> dict[int, dict[str, torch.Tensor]]:
    """
    Runs a forward/backward pass on `transformers_model` using the given `sae`
    for activation compression/decompression. Returns dictionaries keyed by
    layer

    Args:
        transformers_model: A HuggingFace model (e.g., GPT-NeoX).
        sae: An AutoEncoderTopK instance (or similar) that implements
             encode() and decode().
        input_ids: Model inputs, e.g., from tokenizer(..., return_tensors='pt').
        chosen_layers: List of layer indices to register forward hooks on.
        loss_fn: A function that takes model outputs and returns a scalar loss
                 for the backward pass. Defaults to sum of logits.

    Returns:
        A dict of dicts. For each layer index in chosen_layers, we have:
            "grad_x_dot_decoder_BLF": Tensor
            "effects_BLF": Tensor
            "encoded_acts_BLF": Tensor
            "residual_BLD": Tensor
            "x_grad_BLD": Tensor
            "error_effects_BLD": Tensor
    """

    assert len(chosen_layers) == 1, (
        "Only one layer is supported for now. Stop gradients require pass through gradients (See sparse feature circuits appendix)"
    )

    assert padding_side in ["left", "right"]

    # Make sure gradients are enabled for model parameters
    for param in transformers_model.parameters():
        param.requires_grad_(False)

    for param in sae.parameters():
        param.requires_grad_(False)

    # Dict to store the re-encoded layer outputs in the forward hook
    activations = {}

    def save_activation_hook(module, input, output):
        # Get the hidden states (first element of output)
        hidden_states = output[0]
        hidden_states.requires_grad_()  # Re-enable gradient tracking on this detached tensor
        hidden_states.retain_grad()  # Ensure gradients are retained for later inspection
        activations[module] = hidden_states

        return (hidden_states,) + output[1:]

    # def save_activation_hook(module, input, output):
    #     # Get the hidden states (first element of output)
    #     hidden_states = (
    #         output[0].detach().clone()
    #     )  # Detach to cut off gradients from earlier operations
    #     hidden_states.requires_grad_()  # Re-enable gradient tracking on this detached tensor
    #     hidden_states.retain_grad()  # Ensure gradients are retained for later inspection
    #     activations[module] = hidden_states

    #     return (hidden_states,) + output[1:]

    def stop_gradient_activation_hook(module, hook_input, hook_output):
        """
        Forward hook that encodes the hidden states x using `sae`,
        then reconstructs them with x_hat. The difference (residual)
        is detached so it doesn't propagate gradient back through that part.
        """
        x_BLD = hook_output[0]  # hidden states

        with torch.no_grad():
            encoded_acts_BLF = sae.encode(x_BLD)
        x_hat = sae.decode(encoded_acts_BLF)
        residual = x_BLD - x_hat
        # Detach residual so we don't accumulate its gradient
        residual = residual.detach()

        x_recon = x_hat + residual
        # Retain grad so we can do backward on it
        x_recon.requires_grad_()
        x_recon.retain_grad()

        activations[module] = x_recon

        # If the module originally returned a tuple, preserve the structure
        return (x_recon,) + hook_output[1:]

    # Register hooks
    handles = []
    for layer_module in submodules:
        if use_stop_gradient:
            h = layer_module.register_forward_hook(stop_gradient_activation_hook)
        else:
            h = layer_module.register_forward_hook(save_activation_hook)
        handles.append(h)

    if use_activation_loss_fn:
        act_submodule = model_utils.get_submodule(transformers_model, 16)
        h = act_submodule.register_forward_hook(save_activation_hook)
        handles.append(h)

    try:
        # Forward pass
        output_logits_BLV = transformers_model(**model_inputs).logits

        if padding_side == "right":
            raise ValueError(
                "Refactor of loss functions required to support right padding"
            )
            seq_lengths_B = model_inputs["attention_mask"].sum(dim=1) - 1
            answer_logits_B = output_logits_BLV[
                torch.arange(output_logits_BLV.shape[0]),
                seq_lengths_B,
                :,
            ]
        elif padding_side == "left":
            answer_logits_B = output_logits_BLV[
                :,
                -1,
                :,
            ]

        if use_activation_loss_fn:
            loss_fn = make_activation_loss_fn()
            loss = loss_fn(activations[act_submodule], labels)
        else:
            loss = loss_fn(output_logits_BLV, labels)
        loss.backward()

        predicted_tokens = tokenizer.batch_decode(answer_logits_B.argmax(dim=-1))

        if verbose:
            model_name = transformers_model.__class__.__name__
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            view_outputs(tokenizer, model_inputs["input_ids"], answer_logits_B)

    finally:
        # Remove hooks to avoid side effects if function is called repeatedly
        for h in handles:
            h.remove()

    # Now gather the results
    results = {}
    decoder_weight_FD = sae.W_dec.data

    for layer_idx, layer_module in zip(chosen_layers, submodules, strict=True):
        # The post-hook activation
        x_BLD = activations[layer_module]
        x_grad_BLD = x_BLD.grad

        with torch.no_grad():
            encoded_acts_BLF = sae.encode(x_BLD)
            decoded_acts_BLD = sae.decode(encoded_acts_BLF)

            residual_BLD = x_BLD - decoded_acts_BLD

            grad_x_dot_decoder_BLF = torch.einsum(
                "bld,fd->blf", x_grad_BLD, decoder_weight_FD
            )

            if ignore_bos:
                if padding_side == "right":
                    encoded_acts_BLF[:, 0, :] = 0.0
                    residual_BLD[:, 0, :] = 0.0
                elif padding_side == "left":
                    bos_mask = (
                        model_inputs["attention_mask"].cumsum(dim=1) == 1
                    )  # shape [B, L], True where the first 1 occurs
                    # Now zero out the first attended token in each sequence
                    encoded_acts_BLF[bos_mask] = 0.0
                    residual_BLD[bos_mask] = 0.0

            node_effects_BLF = encoded_acts_BLF * grad_x_dot_decoder_BLF * -1
            error_effects_BLD = residual_BLD * x_grad_BLD * -1

            node_effects_BLF *= model_inputs["attention_mask"][:, :, None]
            grad_x_dot_decoder_BLF *= model_inputs["attention_mask"][:, :, None]
            error_effects_BLD *= model_inputs["attention_mask"][:, :, None]
            encoded_acts_BLF *= model_inputs["attention_mask"][:, :, None]

            # Store everything
            results[layer_idx] = {
                "effects_BLF": node_effects_BLF.detach(),
                "grad_x_dot_decoder_BLF": grad_x_dot_decoder_BLF.detach(),
                "encoded_acts_BLF": encoded_acts_BLF.detach(),
                "error_effects_BLD": error_effects_BLD.detach(),
                "labels": labels,
                "model_inputs": model_inputs,
                "predicted_tokens": predicted_tokens,
            }

    return results


def get_effects(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae: base_sae.BaseSAE,
    dataloader: DataLoader,
    yes_vs_no_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    submodules: list[torch.nn.Module],
    chosen_layers: list[int],
    device: torch.device,
    use_activation_loss_fn: bool = False,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list]]:
    effects_F = torch.zeros(sae.W_dec.data.shape[0], device=device)
    error_effect = 0
    predicted_tokens = {"labels": [], "predicted_tokens": []}
    for batch in tqdm(dataloader):
        input_ids, attention_mask, labels, idx_batch = batch

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        batch_results = compute_attributions(
            model,
            tokenizer,
            sae,
            model_inputs,
            labels,
            chosen_layers,
            submodules,
            loss_fn=yes_vs_no_loss_fn,
            # loss_fn=greedy_cross_entropy_loss_fn,
            use_activation_loss_fn=use_activation_loss_fn,
            verbose=False,
            use_stop_gradient=False,
        )

        # Accumulate results from each batch
        with torch.no_grad():
            for layer in chosen_layers:
                effects_BLF = batch_results[layer]["effects_BLF"]
                error_effects_BLD = batch_results[layer]["error_effects_BLD"]

                effects_BLF *= model_inputs["attention_mask"][:, :, None]
                error_effects_BLD *= model_inputs["attention_mask"][:, :, None]

                effects_F += (
                    effects_BLF.to(dtype=torch.float32).sum(dim=(1)).mean(dim=0)
                )
                error_effect += (
                    error_effects_BLD.to(dtype=torch.float32).sum(dim=(1, 2)).mean()
                )
                predicted_tokens["labels"].extend(labels.tolist())
                predicted_tokens["predicted_tokens"].extend(
                    batch_results[layer]["predicted_tokens"]
                )

    effects_F /= len(dataloader)
    error_effect /= len(dataloader)

    stats = analyze_prediction_rates(
        predicted_tokens["labels"],
        predicted_tokens["predicted_tokens"],
        verbose=verbose,
    )

    return effects_F, error_effect, predicted_tokens


@torch.no_grad()
def compute_activations(
    transformers_model: AutoModelForCausalLM,
    sae: base_sae.BaseSAE,
    model_inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    chosen_layers: list[int],
    submodules: list[torch.nn.Module],
    verbose: bool = False,
    ignore_bos: bool = True,
) -> torch.Tensor:
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

    return diff_acts_F


def get_activations(
    model: AutoModelForCausalLM,
    sae: base_sae.BaseSAE,
    dataloader: DataLoader,
    submodules: list[torch.nn.Module],
    chosen_layers: list[int],
    device: torch.device,
) -> torch.Tensor:
    effects_F = torch.zeros(sae.W_dec.data.shape[0], device=device, dtype=torch.float32)

    for batch in tqdm(dataloader):
        input_ids, attention_mask, labels, idx_batch = batch

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        effects_F += compute_activations(
            model,
            sae,
            model_inputs,
            labels,
            chosen_layers,
            submodules,
            verbose=False,
        ).to(dtype=torch.float32)

    effects_F /= len(dataloader)

    return effects_F
