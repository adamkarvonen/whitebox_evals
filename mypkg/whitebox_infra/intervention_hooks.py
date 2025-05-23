import torch
import einops
from jaxtyping import Float
from torch import Tensor
from typing import Callable, Optional
from transformers import AutoTokenizer

import mypkg.whitebox_infra.dictionaries.base_sae as base_sae
import mypkg.pipeline.infra.hiring_bias_prompts as hiring_bias_prompts
import mypkg.whitebox_infra.attribution as attribution


def lookup_sae_features(
    model_name: str,
    trainer_id: int,
    layer_percent: int,
    anti_bias_statement_file: str,
    bias_type: str,
) -> torch.Tensor:
    assert bias_type in ["race", "gender", "political_orientation"]

    anti_bias_statement_file_idx = int(
        anti_bias_statement_file.replace("v", "").replace(".txt", "")
    )

    model_name = model_name.replace("/", "_")

    filename = f"v{anti_bias_statement_file_idx}_trainer_{trainer_id}_model_{model_name}_layer_{layer_percent}_attrib_data.pt"
    attrib_path = f"data/attribution_results_data/{model_name}/{filename}"

    data = torch.load(attrib_path)

    attribution_data = attribution.AttributionData.from_dict(data[bias_type])

    effects_F = attribution_data.pos_effects_F - attribution_data.neg_effects_F

    k = 20

    top_k_ids = effects_F.abs().topk(k).indices

    act_ratios_F = attribution_data.pos_sae_acts_F / attribution_data.neg_sae_acts_F

    top_k_act_ratios = act_ratios_F[top_k_ids]

    adjusted_act_ratios = attribution.adjust_tensor_values(top_k_act_ratios)
    outlier_effect_ids = adjusted_act_ratios > 2.0

    top_k_ids = top_k_ids[outlier_effect_ids]

    return top_k_ids


def get_sae_vectors(
    ablation_features: torch.Tensor, sae: base_sae.BaseSAE
) -> tuple[list[Float[Tensor, "d_model"]], list[Float[Tensor, "d_model"]], list[float]]:
    encoder_vectors = []
    decoder_vectors = []
    encoder_biases = []

    device = sae.W_dec.device

    for i in range(ablation_features.shape[0]):
        feature_idx = ablation_features[i]
        encoder_vector = sae.W_enc[:, feature_idx]
        decoder_vector = sae.W_dec[feature_idx, :]
        encoder_bias = sae.b_enc[feature_idx]

        if hasattr(sae, "threshold") and hasattr(sae, "use_threshold"):
            # batch top k or top k
            encoder_bias += sae.threshold
        elif hasattr(sae, "threshold"):
            encoder_bias += sae.threshold[feature_idx]

        encoder_vector = encoder_vector.to(device)
        decoder_vector = decoder_vector.to(device)
        encoder_bias = encoder_bias.to(device)

        encoder_vectors.append(encoder_vector)
        decoder_vectors.append(decoder_vector)
        encoder_biases.append(encoder_bias)

    return encoder_vectors, decoder_vectors, encoder_biases


def get_conditional_clamping_hook(
    encoder_vectors: list[Float[Tensor, "d_model"]],
    decoder_vectors: list[Float[Tensor, "d_model"]],
    scales: list[float],
    encoder_thresholds: list[float],
) -> Callable:
    """Creates a hook function that conditionally clamps activations.

    Combines conditional intervention with clamping - only clamps activations
    when they exceed the encoder threshold with the decoder intervention.

    Args:
        encoder_vectors: List of vectors defining directions to monitor
        decoder_vectors: List of vectors defining intervention directions
        scales: Target values for clamping
        encoder_thresholds: Threshold values that trigger clamping

    Returns:
        Hook function that conditionally clamps and modifies activations

    Note:
        Most sophisticated intervention type, combining benefits of
        conditional application and activation clamping
    """

    # K = number of encoder/decoder directions
    enc_mat_KD: Float[Tensor, "K d_model"] = torch.stack(
        encoder_vectors
    )  # encoder directions
    dec_mat_KD: Float[Tensor, "K d_model"] = torch.stack(
        decoder_vectors
    )  # decoder directions

    scales_K: Float[Tensor, "K"] = torch.as_tensor(
        scales, dtype=enc_mat_KD.dtype, device=enc_mat_KD.device
    )
    thresholds_K: Float[Tensor, "K"] = torch.as_tensor(
        encoder_thresholds, dtype=enc_mat_KD.dtype, device=enc_mat_KD.device
    )

    def hook_fn(module, _input, output):
        resid_BLD: Float[Tensor, "batch seq_len d_model"] = output[0]

        # Dot product with every encoder dir in one shot -> feats_BLK
        feats_BLK: Float[Tensor, "batch seq_len K"] = torch.einsum(
            "bld,kd->blk", resid_BLD, enc_mat_KD
        )

        # Boolean mask for positions where we clamp
        mask_BLK: torch.BoolTensor = (feats_BLK > thresholds_K) & (feats_BLK > 0)

        # Compute intervention shift Δ_BLKD
        delta_BLKD: Float[Tensor, "batch seq_len K d_model"] = (
            (scales_K - feats_BLK) * mask_BLK
        ).unsqueeze(-1) * dec_mat_KD  # broadcast over d_model

        delta_BLD = einops.reduce(delta_BLKD, "B L K D -> B L D", "sum")

        resid_BLD = resid_BLD + delta_BLD

        return (resid_BLD,) + output[1:]

    return hook_fn


def get_projection_ablation_hook(
    ablate_vectors: list[Float[Tensor, "d_model"]],
) -> Callable:
    """
    Returns a hook that *ablates* (projects out) one or more directions
    from every write to the residual stream.

    Args
    ----
    ablate_vectors : list of direction vectors r_k  (len K)

    Behaviour
    ---------
    For each residual write `resid_BLD`, we compute

        resid_BLD ← resid_BLD  − Σ_k (resid_BLD · r_k) r_k

    so the output tensor is (approximately) orthogonal to every r_k.
    """

    dirs_KD: Float[Tensor, "K d_model"] = torch.stack(ablate_vectors).to(
        dtype=torch.float32
    )

    # Normalize first (optional but good practice)
    dirs_KD = torch.nn.functional.normalize(dirs_KD, dim=1)

    # Orthogonalize the directions using QR decomposition
    # Transpose to (D, K) for QR, then transpose back
    dirs_DK = dirs_KD.T
    Q, R = torch.linalg.qr(dirs_DK, mode="reduced")
    # Q is (D, K) with orthonormal columns
    dirs_KD_ortho = Q.T  # Back to (K, D)

    # Note: If K > D or vectors are linearly dependent, some columns of Q might be zero
    # We should filter those out
    norms = torch.norm(dirs_KD_ortho, dim=1)
    valid_mask = norms > 1e-6
    dirs_KD_ortho = dirs_KD_ortho[valid_mask]

    def hook_fn(module, _input, output):
        resid_BLD: Float[Tensor, "batch seq_len d_model"] = output[0]

        # Dot product with all directions → proj_BLK
        proj_BLK: Float[Tensor, "batch seq_len K"] = torch.einsum(
            "bld,kd->blk", resid_BLD.to(dtype=torch.float32), dirs_KD_ortho
        )

        # Expand back to d_model and subtract
        delta_BLKD: Float[Tensor, "batch seq_len K d_model"] = (
            proj_BLK.unsqueeze(-1) * dirs_KD_ortho
        )
        delta_BLD: Float[Tensor, "batch seq_len d_model"] = delta_BLKD.sum(dim=2)

        resid_BLD = resid_BLD - delta_BLD.to(dtype=resid_BLD.dtype)

        return (resid_BLD,) + output[1:]

    return hook_fn


def get_conditional_steering_hook(
    encoder_vectors: list[Float[Tensor, "d_model"]],
    decoder_vectors: list[Float[Tensor, "d_model"]],
    scales: list[float],
    encoder_thresholds: list[float],
) -> Callable:
    def hook_fn(module, input, output):
        resid_BLD: Float[Tensor, "batch_size seq_len d_model"] = output[0]

        B, L, D = resid_BLD.shape

        for encoder_vector_D, decoder_vector_D, coeff, encoder_threshold in zip(
            encoder_vectors, decoder_vectors, scales, encoder_thresholds
        ):
            # Get encoder activations
            feature_acts_BL = torch.einsum("BLD,D->BL", resid_BLD, encoder_vector_D)

            max_act_B = feature_acts_BL.max(dim=1).values

            # Create mask for where encoder activation exceeds threshold
            intervention_mask_BL = feature_acts_BL > encoder_threshold

            # Calculate clamping amount only where mask is True
            # decoder_BLD = (
            #     feature_acts_BL[:, :, None] * coeff * decoder_vector_D[None, None, :]
            # )

            decoder_BLD = (
                decoder_vector_D[None, None, :] * max_act_B[:, None, None] * coeff
            )

            # Apply clamping only where both mask is True and activation is positive
            resid_BLD = torch.where(
                (intervention_mask_BL[:, :, None]),
                resid_BLD + decoder_BLD,
                resid_BLD,
            )

        return (resid_BLD,) + output[1:]

    return hook_fn


def get_conditional_adaptive_clamping_hook(
    encoder_vectors: list[Float[Tensor, "d_model"]],
    decoder_vectors: list[Float[Tensor, "d_model"]],
    scales: list[float],
    encoder_thresholds: list[float],
) -> Callable:
    """Creates a hook function that conditionally clamps activations.

    Combines conditional intervention with clamping - only clamps activations
    when they exceed the encoder threshold with the decoder intervention.

    Args:
        encoder_vectors: List of vectors defining directions to monitor
        decoder_vectors: List of vectors defining intervention directions
        scales: Target values for clamping
        encoder_thresholds: Threshold values that trigger clamping

    Returns:
        Hook function that conditionally clamps and modifies activations

    Note:
        Most sophisticated intervention type, combining benefits of
        conditional application and activation clamping
    """

    def hook_fn(module, input, output):
        resid_BLD: Float[Tensor, "batch_size seq_len d_model"] = output[0]

        B, L, D = resid_BLD.shape

        for encoder_vector_D, decoder_vector_D, coeff, encoder_threshold in zip(
            encoder_vectors, decoder_vectors, scales, encoder_thresholds
        ):
            # Get encoder activations
            feature_acts_BL = torch.einsum("BLD,D->BL", resid_BLD, encoder_vector_D)

            max_act_B = einops.reduce(feature_acts_BL, "B L -> B", "max")

            # Create mask for where encoder activation exceeds threshold
            intervention_mask_BL = feature_acts_BL > encoder_threshold

            # Calculate clamping amount only where mask is True
            decoder_BLD = (
                -feature_acts_BL[:, :, None] + (coeff * max_act_B[:, None, None])
            ) * decoder_vector_D[None, None, :]

            # Apply clamping only where both mask is True and activation is positive
            resid_BLD = torch.where(
                (intervention_mask_BL[:, :, None] & (feature_acts_BL[:, :, None] > 0)),
                resid_BLD + decoder_BLD,
                resid_BLD,
            )

        return (resid_BLD,) + output[1:]

    return hook_fn


def get_conditional_adaptive_steering_hook(
    encoder_vectors: list[Float[Tensor, "d_model"]],
    decoder_vectors: list[Float[Tensor, "d_model"]],
    scales: list[float],
    encoder_thresholds: list[float],
) -> Callable:
    """Creates a hook function that conditionally clamps activations.

    Combines conditional intervention with clamping - only clamps activations
    when they exceed the encoder threshold with the decoder intervention.

    Args:
        encoder_vectors: List of vectors defining directions to monitor
        decoder_vectors: List of vectors defining intervention directions
        scales: Target values for clamping
        encoder_thresholds: Threshold values that trigger clamping

    Returns:
        Hook function that conditionally clamps and modifies activations

    Note:
        Most sophisticated intervention type, combining benefits of
        conditional application and activation clamping
    """

    def hook_fn(module, input, output):
        resid_BLD: Float[Tensor, "batch_size seq_len d_model"] = output[0]

        B, L, D = resid_BLD.shape

        for encoder_vector_D, decoder_vector_D, coeff, encoder_threshold in zip(
            encoder_vectors, decoder_vectors, scales, encoder_thresholds
        ):
            # Get encoder activations
            feature_acts_BL = torch.einsum("BLD,D->BL", resid_BLD, encoder_vector_D)

            max_act_B = einops.reduce(feature_acts_BL, "B L -> B", "max")

            # Create mask for where encoder activation exceeds threshold
            intervention_mask_BL = feature_acts_BL > encoder_threshold

            # Calculate clamping amount only where mask is True
            decoder_BLD = (
                feature_acts_BL[:, :, None] + (coeff * max_act_B[:, None, None])
            ) * decoder_vector_D[None, None, :]

            # Apply clamping only where both mask is True and activation is positive
            resid_BLD = torch.where(
                (intervention_mask_BL[:, :, None] & (feature_acts_BL[:, :, None] > 0)),
                resid_BLD + decoder_BLD,
                resid_BLD,
            )

        return (resid_BLD,) + output[1:]

    return hook_fn


def get_constant_steering_hook(
    encoder_vectors: list[Float[Tensor, "d_model"]],
    decoder_vectors: list[Float[Tensor, "d_model"]],
    scales: list[float],
    encoder_thresholds: list[float],
) -> Callable:
    def hook_fn(module, input, output):
        resid_BLD: Float[Tensor, "batch_size seq_len d_model"] = output[0]

        B, L, D = resid_BLD.shape

        for encoder_vector_D, decoder_vector_D, coeff, encoder_threshold in zip(
            encoder_vectors, decoder_vectors, scales, encoder_thresholds
        ):
            # Get encoder activations
            feature_acts_BL = torch.einsum("BLD,D->BL", resid_BLD, encoder_vector_D)

            # Create mask for where encoder activation exceeds threshold
            intervention_mask_BL = feature_acts_BL > encoder_threshold

            # Calculate clamping amount only where mask is True
            decoder_BLD = coeff * decoder_vector_D[None, None, :]

            # Apply clamping only where both mask is True and activation is positive
            resid_BLD = resid_BLD + decoder_BLD

        return (resid_BLD,) + output[1:]

    return hook_fn


def get_targeted_steering_hook(
    encoder_vectors: list[Float[Tensor, "d_model"]],
    decoder_vectors: list[Float[Tensor, "d_model"]],
    scales: list[float],
    encoder_thresholds: list[float],
    resume_prompt_results: list[hiring_bias_prompts.ResumePromptResult],
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
) -> Callable:
    resume_mask_BL = torch.zeros_like(input_ids)

    # Note: This is currently pretty slow, could be sped up more by precomputing the encoded suffix and resume
    for i, resume_prompt in enumerate(resume_prompt_results):
        resume = resume_prompt.resume
        prompt_splits = resume_prompt.prompt.split(resume)
        assert len(prompt_splits) == 2
        suffix = prompt_splits[-1]
        encoded_suffix = tokenizer.encode(suffix, add_special_tokens=False)
        encoded_resume = tokenizer.encode(resume, add_special_tokens=False)
        resume_end = -len(encoded_suffix)
        resume_start = resume_end - len(encoded_resume)
        resume_mask_BL[i, resume_start:resume_end] = 1

    def hook_fn(module, input, output):
        resid_BLD: Float[Tensor, "batch_size seq_len d_model"] = output[0]

        B, L, D = resid_BLD.shape

        for encoder_vector_D, decoder_vector_D, coeff, encoder_threshold in zip(
            encoder_vectors, decoder_vectors, scales, encoder_thresholds
        ):
            # Get encoder activations
            feature_acts_BL = torch.einsum("BLD,D->BL", resid_BLD, encoder_vector_D)

            # Create mask for where encoder activation exceeds threshold
            intervention_mask_BL = feature_acts_BL > encoder_threshold

            # Calculate clamping amount only where mask is True
            decoder_BLD = coeff * decoder_vector_D[None, None, :]

            decoder_BLD = decoder_BLD * resume_mask_BL[:, :, None]

            # Apply clamping only where both mask is True and activation is positive
            resid_BLD = resid_BLD + decoder_BLD

        return (resid_BLD,) + output[1:]

    return hook_fn


def get_ablation_hook(
    ablation_type: str,
    encoder_vectors: list[Float[Tensor, "d_model"]],
    decoder_vectors: list[Float[Tensor, "d_model"]],
    scales: list[float],
    encoder_biases: list[float],
    resume_prompt_results_batch: Optional[
        list[hiring_bias_prompts.ResumePromptResult]
    ] = None,
    input_ids: Optional[torch.Tensor] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> Callable:
    pass

    if ablation_type == "clamping":
        ablation_hook = get_conditional_clamping_hook(
            encoder_vectors, decoder_vectors, scales, encoder_biases
        )
    elif ablation_type == "steering":
        ablation_hook = get_conditional_steering_hook(
            encoder_vectors, decoder_vectors, scales, encoder_biases
        )
    elif ablation_type == "constant":
        ablation_hook = get_constant_steering_hook(
            encoder_vectors, decoder_vectors, scales, encoder_biases
        )
    elif ablation_type == "adaptive_clamping":
        ablation_hook = get_conditional_adaptive_clamping_hook(
            encoder_vectors, decoder_vectors, scales, encoder_biases
        )
    elif ablation_type == "adaptive_steering":
        ablation_hook = get_conditional_adaptive_steering_hook(
            encoder_vectors, decoder_vectors, scales, encoder_biases
        )
    elif ablation_type == "targeted":
        ablation_hook = get_targeted_steering_hook(
            encoder_vectors,
            decoder_vectors,
            scales,
            encoder_biases,
            resume_prompt_results_batch,
            input_ids,
            tokenizer,
        )
    elif ablation_type == "projection_ablations":
        ablation_hook = get_projection_ablation_hook(encoder_vectors)
    else:
        raise ValueError(f"Invalid ablation type: {ablation_type}")

    return ablation_hook
