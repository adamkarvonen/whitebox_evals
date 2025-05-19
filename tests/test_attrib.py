import pytest
import argparse
import asyncio
import numbers
import os
import random
import numpy as np
import torch

import attribution_experiment


def test_attrib():
    top_20_indices = torch.tensor(
        [
            12553,
            10565,
            2841,
            2908,
            6482,
            12263,
            534,
            2373,
            11640,
            6500,
            3235,
            5038,
            14392,
            13621,
            697,
            10859,
            13324,
            15356,
            2311,
            8286,
        ],
        device="cpu",
    )
    top_20_vals = torch.tensor(
        [
            -0.0276,
            -0.0184,
            -0.0154,
            -0.0129,
            -0.0128,
            0.0128,
            0.0122,
            0.0112,
            0.0111,
            0.0102,
            0.0100,
            -0.0096,
            0.0085,
            -0.0085,
            -0.0083,
            0.0078,
            0.0077,
            -0.0073,
            0.0070,
            0.0068,
        ],
        device="cpu",
    )
    error_effect = torch.tensor(3.273040056228638e-05, device="cpu")

    # 1. Turn off TF32 so matmul precision matches across GPUs
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False  # PyTorch ≥2.1
    torch.set_float32_matmul_precision("high")

    # 2. Keep your existing strict flags
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # still needed on Hopper

    # 3. Seed once per process
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    EXPECTED_RESULTS = {
        "top_20_indices": top_20_indices,
        "top_20_vals": top_20_vals,
        "error_effect": error_effect,
    }

    bias_category = "gender"

    args = argparse.Namespace(
        model_name="google/gemma-2-2b-it",
        anti_bias_statement_file="v1.txt",
        downsample=5,
        output_dir="test_attribution_results",
        chosen_layer_percentage=25,
    )

    all_test_data = attribution_experiment.main(
        args, bias_categories_to_test=[bias_category], override_trainer_id=16
    )

    test_data = all_test_data[bias_category]

    effects_F = test_data["pos_effects_F"] - test_data["neg_effects_F"]
    error_effect = test_data["pos_error_effect"] - test_data["neg_error_effect"]
    error_effect = torch.tensor(error_effect, device="cpu")

    top_k_ids = effects_F.abs().topk(20).indices
    top_k_vals = effects_F[top_k_ids]

    assert torch.allclose(top_k_ids, EXPECTED_RESULTS["top_20_indices"])
    assert torch.allclose(top_k_vals, EXPECTED_RESULTS["top_20_vals"], atol=1e-4)
    assert torch.allclose(error_effect, EXPECTED_RESULTS["error_effect"], atol=1e-4)
