import pytest
import argparse
import asyncio
import numbers
import os
import random
import numpy as np
import torch

import attribution_experiment


def test_forward_pass():
    top_20_indices = torch.tensor(
        [
            2870,
            12553,
            6500,
            10565,
            534,
            13789,
            2908,
            2328,
            6482,
            10597,
            12263,
            2311,
            619,
            14392,
            11640,
            5038,
            11241,
            697,
            14217,
            3235,
        ],
        device="cpu",
    )
    top_20_vals = torch.tensor(
        [
            -1.1457,
            -0.8101,
            0.7938,
            -0.7884,
            0.7651,
            0.6948,
            0.6771,
            -0.6226,
            -0.5178,
            0.4570,
            0.4533,
            0.4385,
            -0.3777,
            0.3737,
            0.3724,
            -0.3704,
            -0.3497,
            -0.3200,
            0.3123,
            0.3114,
        ],
        device="cpu",
    )
    error_effect = torch.tensor(1.4059, device="cpu")

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
        output_dir="attribution_results",
    )

    all_test_data = attribution_experiment.main(
        args, bias_categories_to_test=[bias_category]
    )

    test_data = all_test_data[bias_category]

    effects_F = test_data["effects_F"]
    error_effect = test_data["error_effect"]

    top_k_ids = effects_F.abs().topk(20).indices
    top_k_vals = effects_F[top_k_ids]

    assert torch.allclose(top_k_ids, EXPECTED_RESULTS["top_20_indices"])
    assert torch.allclose(top_k_vals, EXPECTED_RESULTS["top_20_vals"], atol=1e-4)
    assert torch.allclose(error_effect, EXPECTED_RESULTS["error_effect"], atol=1e-4)
