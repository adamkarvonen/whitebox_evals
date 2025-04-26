import os

# Must set before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import pandas as pd
from tqdm import tqdm
import sys
import os
from datetime import datetime
from tabulate import tabulate
import asyncio
import json
import random
from datasets import load_dataset
import torch
from dataclasses import asdict
import itertools
import gc
import time
from typing import Optional, Any
from enum import Enum

from mypkg.eval_config import EvalConfig
from mypkg.pipeline.setup.dataset import (
    load_raw_dataset,
    filter_by_industry,
    filter_by_demographics,
    prepare_dataset_for_model,
)
from mypkg.pipeline.infra.hiring_bias_prompts import (
    create_all_prompts_hiring_bias,
    create_all_prompts_anthropic,
    evaluate_bias,
    evaluate_bias_probs,
    filter_anthropic_df,
    modify_anthropic_filled_templates,
)
import mypkg.pipeline.infra.model_inference as model_inference
import mypkg.whitebox_infra.model_utils as model_utils
import mypkg.whitebox_infra.intervention_hooks as intervention_hooks


# Define the InferenceMode Enum
class InferenceMode(Enum):
    GPU_INFERENCE = "gpu_inference"
    GPU_FORWARD_PASS = "gpu_forward_pass"
    PERFORM_ABLATIONS = "perform_ablations"
    OPEN_ROUTER = "open_router"

    def __str__(self):
        return self.value


OPENROUTER_NAME_LOOKUP = {
    "mistralai/Ministral-8B-Instruct-2410": "mistralai/ministral-8b",
    "mistralai/Mistral-Small-24B-Instruct-2501": "mistralai/mistral-small-3.1-24b-instruct",
}


# Moved Logger class to top level
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def save_evaluation_results(
    args,
    model_name: str,
    df,
    results,
    bias_scores,
    cache_dir: str,
    timestamp: str,
    eval_config: EvalConfig,
):
    """
    Saves a comprehensive record of this run as a single JSON.
    Includes:
      - metadata (timestamp, model_name, arguments)
      - all prompt/response details
      - bias summary
      - any additional summary info (like # of resumes processed, industries, etc.)
    """
    # Build a big dictionary with all data:
    # 1) Metadata
    evaluation_data = {
        "metadata": {
            "timestamp": timestamp,
            "model_name": model_name,
            # Convert argparse.Namespace to a dict so it's JSON-serializable
            "arguments": vars(args),
        },
        # 2) Full list of results
        "results": [],
        # 3) Bias summary
        "bias_summary": bias_scores,
        # 4) Additional info
        "summary_info": {
            "total_resumes_processed": len(results),
            "industries_included": list(df["Category"].unique()),
        },
        "eval_config": asdict(eval_config),
    }

    # For each ResumePromptResult in `results`, store relevant info
    # so you can fully reconstruct or debug each inference later.
    for item in results:
        evaluation_data["results"].append(asdict(item))

    # Use a filename that includes the timestamp to keep files distinct
    output_filename = f"evaluation_{timestamp}_{model_name}_{eval_config.anti_bias_statement_file}_{eval_config.job_description_file}.json"
    output_filename = output_filename.replace(".txt", "").replace("/", "_")
    output_path = os.path.join(cache_dir, output_filename)

    # Write the dictionary to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=4)

    print(f"\nSaved evaluation data to: {output_path}")


def balanced_downsample(df, n_samples, random_seed=42):
    """
    Downsample the dataset while maintaining balance across resume content.

    Args:
        df: DataFrame containing the dataset
        n_samples: Number of unique resume contents to sample
        random_seed: Random seed for reproducibility

    Returns:
        Downsampled DataFrame that maintains demographic variations for each resume
    """
    # Get unique resume strings
    unique_resumes = df["Resume_str"].unique()

    if n_samples > len(unique_resumes):
        print(
            f"Warning: Requested sample size ({n_samples}) is larger than unique resumes ({len(unique_resumes)}). Using all unique resumes."
        )
        return df

    # Randomly sample n_samples unique resumes
    sampled_resumes = random.sample(list(unique_resumes), n_samples)

    # Get all rows that contain these resume strings
    balanced_sample = df[df["Resume_str"].isin(sampled_resumes)]

    print(f"Downsampled to {len(sampled_resumes)} unique resumes")
    print(
        f"Total samples after maintaining demographic variations: {len(balanced_sample)}"
    )

    return balanced_sample


REASONING_MODELS = [
    "openai/o1-mini",
    "x-ai/grok-3-mini-beta",
    "qwen/qwq-32b",
    "openai/o1",
    "anthropic/claude-3.7-sonnet:thinking",
    "deepseek/deepseek-r1",
]


# Moved argument parsing to a separate function
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--industry",
        type=str,
        help="Industry to evaluate (optional)",
        default="INFORMATION-TECHNOLOGY",
    )
    parser.add_argument("--mode", type=str, choices=["full", "summary"], default="full")
    parser.add_argument(
        "--political_orientation",
        action="store_true",
        help="Whether to include political orientation information",
    )
    parser.add_argument(
        "--pregnancy",
        action="store_true",
        help="Whether to include pregnancy information",
    )
    parser.add_argument(
        "--employment_gap",
        action="store_true",
        help="Whether to include employment gap information",
    )
    parser.add_argument(
        "--misc",
        action="store_true",
        help="Whether to include misc information",
    )
    parser.add_argument(
        "--anthropic_dataset",
        action="store_true",
        help="Whether to use the Anthropic dataset",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Number of samples to randomly select from the dataset",
    )
    parser.add_argument(
        "--system_prompt_filename",
        type=str,
        default="yes_no.txt",
        help="System prompt filename to use",
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=[mode.value for mode in InferenceMode],
        default=InferenceMode.OPEN_ROUTER.value,
        help="The inference mode to use.",
    )
    parser.add_argument(
        "--anti_bias_statement_file",
        type=str,
        default=None,
        help="Path to a single anti-bias statement file; We only use this statement.",
    )
    parser.add_argument(
        "--score_output_dir",
        type=str,
        default="score_output",
        help="Path to a directory to save the score output.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="The model name to use. If not provided, we will use all model_names in the model_names list.",
    )
    parser.add_argument(
        "--bias_type",
        type=str,
        default=None,
        help="The bias type to use. If not provided, we will use all bias_types in the bias_types list.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="The scale to use. If not provided, we will use all scales in the scales list.",
    )
    parser.add_argument(
        "--overwrite_existing_results",
        action="store_true",
        help="Whether to overwrite existing results",
    )

    args = parser.parse_args()

    return args


async def main(
    args: argparse.Namespace,
    cache_dir: str,
    timestamp: str,
) -> dict[str, Any]:
    """python mypkg/main_paper_dataset.py --downsample 20 --system_prompt_filename yes_no_cot.txt --inference_mode gpu_inference

    python mypkg/main_paper_dataset.py --downsample 20 --system_prompt_filename yes_no.txt --inference_mode gpu_forward_pass

    python mypkg/main_paper_dataset.py --downsample 50 --system_prompt_filename yes_no.txt --inference_mode perform_ablations

    python mypkg/main_paper_dataset.py --downsample 50 --system_prompt_filename yes_no.txt"""
    os.makedirs(cache_dir, exist_ok=True)

    start_time = time.time()

    if sum([args.political_orientation, args.pregnancy, args.employment_gap]) > 1:
        raise ValueError(
            "Only one of political orientation, pregnancy, or employment gap can be true"
        )

    print("Loading dataset...")
    if args.anthropic_dataset:
        # TODO: Experiment with explicit vs implicit
        dataset = load_dataset("Anthropic/discrim-eval", "explicit")
        # dataset = load_dataset("Anthropic/discrim-eval", "implicit")
        df = dataset["train"]
        df = filter_anthropic_df(df)
        df = modify_anthropic_filled_templates(df)
    else:
        df = load_raw_dataset()

        if args.industry:
            print(f"Filtering for industry: {args.industry}")
            df = filter_by_industry(df, args.industry)
        else:
            print("No industry filter applied.")

    eval_config = EvalConfig(
        inference_mode=args.inference_mode,
        model_name="",
        industry=args.industry,
        mode=args.mode,
        political_orientation=args.political_orientation,
        pregnancy=args.pregnancy,
        employment_gap=args.employment_gap,
        anthropic_dataset=args.anthropic_dataset,
        downsample=args.downsample,
        system_prompt_filename=args.system_prompt_filename,
        anti_bias_statement_file=args.anti_bias_statement_file,
    )

    # Set the seed for reproducibility
    random_seed = eval_config.random_seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    if args.downsample is not None:
        if args.anthropic_dataset:
            print("\n\n\n\nDownsampling Anthropic dataset is not supported\n\n\n")
        else:
            df = balanced_downsample(df, args.downsample, eval_config.random_seed)

    # Print demographic distribution before sampling
    print("\n[FULL DATASET:] Demographic distribution:")
    print("Gender:", df["Gender"].value_counts())
    print("--------------------------------")
    print("Race:", df["Race"].value_counts())
    print("--------------------------------")

    # job_descriptions = ["meta_job_description.txt", "short_meta_job_description.txt"]
    job_descriptions = ["short_meta_job_description.txt"]

    if args.anti_bias_statement_file is None:
        anti_bias_statement_files = [f"v{i}.txt" for i in range(0, 18)]
    else:
        anti_bias_statement_files = [args.anti_bias_statement_file]

    if args.model_name is None:
        model_names = [
            "google/gemma-2-2b-it",
            # "google/gemma-2-9b-it",
            # "google/gemma-2-27b-it",
            # "mistralai/Ministral-8B-Instruct-2410",
            # "mistralai/Mistral-Small-24B-Instruct-2501",
            # "deepseek/deepseek-r1",
            # "openai/gpt-4o-2024-08-06",
            # "deepseek/deepseek-r1-distill-llama-70b"
            # "openai/o1-mini-2024-09-12",
            # "openai/o1-mini",
            # "openai/o1"
            # "x-ai/grok-3-mini-beta"
            # "qwen/qwq-32b",
            # "anthropic/claude-3.7-sonnet"
            # "anthropic/claude-3.7-sonnet:thinking",
            # "qwen/qwen2.5-32b-instruct",
            # "openai/gpt-4o-mini",
        ]
    else:
        model_names = [args.model_name]

    os.makedirs(args.score_output_dir, exist_ok=True)

    if (
        eval_config.system_prompt_filename == "yes_no_cot.txt"
        or eval_config.system_prompt_filename == "yes_no_qualifications.txt"
    ):
        max_completion_tokens = 200
    else:
        max_completion_tokens = 5

    if model_names[0] in REASONING_MODELS:
        assert len(model_names) == 1
        max_completion_tokens = None

    max_completion_tokens = None

    if args.inference_mode == InferenceMode.GPU_INFERENCE.value:
        # We load this here because it often takes ~1 minute to load
        import vllm

        assert len(model_names) == 1
        vllm_model = vllm.LLM(model=model_names[0], dtype="bfloat16")

    # Determine scales and bias_types based on inference_mode
    if args.inference_mode == InferenceMode.PERFORM_ABLATIONS.value:
        scales = [0.0, 1.0, 2.0]
        bias_types = ["gender", "race", "political_orientation"]

        # override bias_types and scales if provided
        if args.bias_type is not None:
            bias_types = [args.bias_type]
        if args.scale is not None:
            scales = [args.scale]
    else:
        scales = [0.0]
        bias_types = ["N/A"]

    all_results = {}

    for (
        job_description,
        model_name,
        anti_bias_statement_file,
        scale,
        bias_type,
    ) in itertools.product(
        job_descriptions, model_names, anti_bias_statement_files, scales, bias_types
    ):
        eval_config.anti_bias_statement_file = anti_bias_statement_file
        eval_config.job_description_file = job_description
        eval_config.model_name = model_name
        eval_config.scale = scale
        eval_config.bias_type = bias_type
        print(f"Running with anti-bias statement: {anti_bias_statement_file}")
        print(f"Running with job description: {job_description}")
        print(f"Running with model: {model_name}")
        print(f"Running with scale: {scale}")
        print(f"Running with bias type: {bias_type}")

        if args.inference_mode == InferenceMode.PERFORM_ABLATIONS.value:
            if bias_type == "political_orientation":
                args.political_orientation = True
                eval_config.political_orientation = True
            elif bias_type == "gender" or bias_type == "race":
                args.political_orientation = False
                eval_config.political_orientation = False
            else:
                raise ValueError(f"Unhandled bias type: {bias_type}")

        temp_results_filename = f"score_results_{anti_bias_statement_file}_{job_description}_{model_name}_{str(scale).replace('.', '_')}_{bias_type}.json".replace(
            ".txt", ""
        ).replace("/", "_")
        temp_results_folder = os.path.join(
            args.inference_mode, model_name.replace("/", "_")
        )
        output_dir = os.path.join(args.score_output_dir, temp_results_folder)
        os.makedirs(output_dir, exist_ok=True)

        file_key = os.path.join(temp_results_folder, temp_results_filename)
        temp_results_filepath = os.path.join(output_dir, temp_results_filename)

        if (
            os.path.exists(temp_results_filepath)
            and not args.overwrite_existing_results
        ):
            print(f"Skipping {temp_results_filename} because it already exists")
            continue

        if args.anthropic_dataset:
            prompts = create_all_prompts_anthropic(df, args, eval_config)
        else:
            prompts = create_all_prompts_hiring_bias(df, args, eval_config)

        gc.collect()
        torch.cuda.empty_cache()

        if args.inference_mode == InferenceMode.GPU_FORWARD_PASS.value:
            batch_size = model_utils.MODEL_CONFIGS[model_name]["batch_size"]
            results = model_inference.run_single_forward_pass_transformers(
                prompts, model_name, batch_size=batch_size
            )
        elif args.inference_mode == InferenceMode.PERFORM_ABLATIONS.value:
            batch_size = model_utils.MODEL_CONFIGS[model_name]["batch_size"]

            ablation_features = intervention_hooks.lookup_sae_features(
                model_name,
                model_utils.MODEL_CONFIGS[model_name]["trainer_id"],
                25,
                anti_bias_statement_file,
                bias_type,
            )

            print(ablation_features)

            results = model_inference.run_single_forward_pass_transformers(
                prompts,
                model_name,
                batch_size=batch_size,
                ablation_features=ablation_features,
                ablation_type="adaptive_clamping",
                scale=scale,
            )

        elif args.inference_mode == InferenceMode.GPU_INFERENCE.value:
            results = model_inference.run_inference_vllm(
                prompts,
                model_name,
                max_new_tokens=max_completion_tokens,
                model=vllm_model,
            )
        elif args.inference_mode == InferenceMode.OPEN_ROUTER.value:
            # Use the lookup table if the model name is present
            api_model_name = OPENROUTER_NAME_LOOKUP.get(model_name, model_name)
            results = await model_inference.run_model_inference_openrouter(
                prompts, api_model_name, max_completion_tokens=max_completion_tokens
            )
        else:
            # This case should not be reachable due to argparse choices
            raise ValueError(f"Unhandled inference mode: {args.inference_mode}")

        # Quick and dirty check to see if prompts are the same from run to run
        total_len = 0
        for result in results:
            total_len += len(result.prompt)
        print(f"Total length of prompts: {total_len}")

        bias_scores = evaluate_bias(
            results, system_prompt_filename=args.system_prompt_filename
        )
        print(bias_scores)

        run_results = {
            "bias_scores": bias_scores,
            "model_name": eval_config.model_name,
            "anti_bias_statement": args.anti_bias_statement_file,
            "job_description": job_description,
        }

        if args.inference_mode in [
            InferenceMode.GPU_FORWARD_PASS.value,
            InferenceMode.PERFORM_ABLATIONS.value,
        ]:
            bias_probs = evaluate_bias_probs(results)
            print("\n\n\n", bias_probs)
            run_results["bias_probs"] = bias_probs

        save_evaluation_results(
            args,
            model_name,
            df,
            results,
            bias_scores,
            cache_dir,
            timestamp,
            eval_config,
        )
        # industries list info
        print(f"\nTotal resumes processed: {len(results)}")
        print(f"Industries included: {', '.join(df['Category'].unique())}")

        all_results[file_key] = run_results

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
            print(f"Peak CUDA memory usage: {peak_memory:.2f} MB")

        with open(temp_results_filepath, "w") as f:
            json.dump(run_results, f)

        end_time = time.time()
        print(f"Total time taken: {end_time - start_time:.2f} seconds")

    return all_results  # Return the collected results


if __name__ == "__main__":
    # Setup moved here
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    log_file = os.path.join(cache_dir, f"history_output_{timestamp}.txt")
    original_stdout = sys.stdout  # Keep track of the original stdout
    sys.stdout = Logger(log_file)

    try:
        args = parse_args()
        # Pass args, cache_dir, and timestamp to main
        results = asyncio.run(main(args, cache_dir, timestamp))
        # Optionally print or process returned results here
        # print("Final results dictionary returned by main:")
        # print(json.dumps(results, indent=4))
    finally:
        # Ensure stdout is reset and logger is closed even if errors occur
        sys.stdout.log.close()
        sys.stdout = original_stdout  # Restore original stdout
