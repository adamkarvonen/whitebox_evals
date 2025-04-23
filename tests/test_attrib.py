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
            2870,
            12553,
            534,
            6500,
            2908,
            10565,
            6482,
            2328,
            12263,
            14217,
            10597,
            2841,
            697,
            2311,
            14392,
            11640,
            11241,
            5038,
            3235,
            4510,
        ],
        device="cpu",
    )
    top_20_vals = torch.tensor(
        [
            -0.0264,
            -0.0204,
            0.0184,
            0.0176,
            0.0160,
            -0.0144,
            -0.0127,
            -0.0120,
            0.0118,
            0.0117,
            0.0102,
            -0.0086,
            -0.0082,
            0.0078,
            0.0076,
            0.0075,
            -0.0072,
            -0.0069,
            0.0067,
            -0.0065,
        ],
        device="cpu",
    )
    error_effect = torch.tensor(1.5813857316970825e-05, device="cpu")

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
    )

    all_test_data = attribution_experiment.main(
        args, bias_categories_to_test=[bias_category]
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
