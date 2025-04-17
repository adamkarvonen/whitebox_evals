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
import vllm

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
    filter_anthropic_df,
    modify_anthropic_filled_templates,
)
import mypkg.pipeline.infra.model_inference as model_inference
import mypkg.whitebox_infra.model_utils as model_utils

OPENROUTER_NAME_LOOKUP = {
    "mistralai/Ministral-8B-Instruct-2410": "mistralai/ministral-8b",
    "mistralai/Mistral-Small-24B-Instruct-2501": "mistralai/mistral-small-3.1-24b-instruct",
}


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
        evaluation_data["results"].append(
            {
                "name": item.name,
                "gender": item.gender,
                "race": item.race,
                "politics": item.politics,
                "job_category": item.job_category,
                "pregnancy_added": item.pregnancy_added,
                "employment_gap_added": item.employment_gap_added,
                "political_orientation_added": item.political_orientation_added,
                "system_prompt": item.system_prompt,
                "task_prompt": item.task_prompt,
                "prompt_text": item.prompt,
                "response": item.response,
            }
        )

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


model_features = {
    "mistralai/Ministral-8B-Instruct-2410": [
        4794,
        4393,
        3645,
        15242,
        2039,
        9049,
        11802,
        13855,
        5286,
        4204,
        428,
    ]
}


async def main():
    """python mypkg/main_paper_dataset.py --downsample 20 --system_prompt_filename yes_no_cot.txt --anti_bias_statement_file v1.txt --gpu_inference

    python mypkg/vllm_inference.py --gpu_inference --vllm_model_name mistralai/Mistral-Small-24B-Instruct-2501 --political_orientation ;
    python mypkg/vllm_inference.py --gpu_inference --vllm_model_name mistralai/Ministral-8B-Instruct-2410 --political_orientation ;
    python mypkg/vllm_inference.py --gpu_inference --vllm_model_name google/gemma-2-9b-it --political_orientation

    python mypkg/vllm_inference.py --downsample 20 --system_prompt_filename yes_no.txt --gpu_forward_pass --vllm_model_name google/gemma-2-2b-it

    python mypkg/main_paper_dataset.py --downsample 20 --system_prompt_filename yes_no.txt --anti_bias_statement_file v1.txt --gpu_forward_pass

    python mypkg/main_paper_dataset.py --downsample 50 --system_prompt_filename yes_no.txt --anti_bias_statement_file v1.txt"""

    # model_names = [
    #     # "google/gemma-2-9b-it",
    #     # "google/gemma-2-27b-it",
    #     # "mistralai/Ministral-8B-Instruct-2410",
    #     "mistralai/Mistral-Small-24B-Instruct-2501",
    # ]

    # Set up logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    log_file = os.path.join(cache_dir, f"history_output_{timestamp}.txt")

    start_time = time.time()

    # Create a custom logger that writes to both file and console
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

    sys.stdout = Logger(log_file)

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
        "--gpu_inference",
        action="store_true",
        help="Whether to use GPU inference",
    )
    parser.add_argument(
        "--gpu_forward_pass",
        action="store_true",
        help="Whether to use GPU forward pass",
    )
    parser.add_argument(
        "--perform_ablations",
        action="store_true",
        help="Whether to perform ablation experiments",
    )
    parser.add_argument(
        "--overwrite_existing_results",
        action="store_true",
        help="Whether to overwrite existing results",
    )
    parser.add_argument(
        "--anti_bias_statement_file",
        type=str,
        required=False,
        default=None,
        help="Path to a single anti-bias statement file; We only use this statement.",
    )
    parser.add_argument(
        "--vllm_model_name",
        type=str,
        required=True,
        help="VLLM model name to use",
    )
    parser.add_argument(
        "--score_output_dir",
        type=str,
        default="score_output",
        help="Path to a directory to save the score output.",
    )

    args = parser.parse_args()

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

    # model_name = "gpt-4o-mini"
    # model_name = "google/gemma-2-9b-it"
    # model_name = "google/gemma-2-27b-it"
    # model_name = "qwen/qwen2.5-32b-instruct"
    # model_name = "deepseek/deepseek-r1"
    # model_name = "qwen/qwen-2.5-7b-instruct"
    # model_name = "google/gemini-flash-1.5-8b"
    # model_name = "meta-llama/llama-3.1-8b-instruct"
    # model_name = "mistralai/Ministral-8B-Instruct-2410"
    # model_name = "mistralai/mixtral-8x7b-instruct"
    # model_name = "mistralai/mistral-small-3.1-24b-instruct"

    model_name = args.vllm_model_name

    eval_config = EvalConfig(
        model_name=model_name,
        industry=args.industry,
        mode=args.mode,
        political_orientation=args.political_orientation,
        pregnancy=args.pregnancy,
        employment_gap=args.employment_gap,
        anthropic_dataset=args.anthropic_dataset,
        downsample=args.downsample,
        gpu_inference=args.gpu_inference,
        system_prompt_filename=args.system_prompt_filename,
        anti_bias_statement_file=args.anti_bias_statement_file,
    )
    print(f"Running with model: {model_name}")

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

    job_descriptions = ["meta_job_description.txt", "short_meta_job_description.txt"]
    # job_descriptions = ["meta_job_description.txt"]
    # job_descriptions = ["long_meta_job_description_v2.txt"]
    # job_descriptions = ["short_meta_job_description.txt"]
    # model_names = ["mistralai/Ministral-8B-Instruct-2410"]
    # model_names = ["mistralai/Mistral-Small-24B-Instruct-2501"]

    anti_bias_statement_files = [f"v{i}.txt" for i in range(0, 18)]

    os.makedirs(args.score_output_dir, exist_ok=True)

    if eval_config.system_prompt_filename == "yes_no_cot.txt":
        max_completion_tokens = 200
    else:
        max_completion_tokens = 5

    # Make sure the chat template exists
    chat_template_test = model_utils.add_chat_template(["Test"], model_name)

    if args.gpu_inference:
        model = vllm.LLM(model=model_name, dtype="bfloat16")

    for anti_bias_statement_file in anti_bias_statement_files:
        print(f"Running with anti-bias statement: {anti_bias_statement_file}")
        eval_config.anti_bias_statement_file = anti_bias_statement_file

        temp_results_filename = (
            f"score_results_{anti_bias_statement_file}.json".replace(
                ".txt", ""
            ).replace("/", "_")
        )
        output_dir = os.path.join(args.score_output_dir, model_name.replace("/", "_"))
        os.makedirs(output_dir, exist_ok=True)
        temp_results_filepath = os.path.join(output_dir, temp_results_filename)

        temp_results = {}

        if os.path.exists(temp_results_filepath):
            if not args.overwrite_existing_results:
                print(f"Skipping {temp_results_filepath} because it already exists")
                continue
            else:
                print(f"Deleting {temp_results_filepath}")
                os.remove(temp_results_filepath)

        for job_description in job_descriptions:
            eval_config.job_description_file = job_description
            print(f"Running with job description: {job_description}")

            if eval_config.anthropic_dataset:
                prompts = create_all_prompts_anthropic(df, args, eval_config)
            else:
                prompts = create_all_prompts_hiring_bias(df, args, eval_config)

            gc.collect()
            torch.cuda.empty_cache()

            if args.gpu_forward_pass:
                # raise NotImplementedError("GPU forward pass not implemented for VLLM")
                batch_size = model_utils.MODEL_CONFIGS[model_name]["batch_size"]
                results = model_inference.run_single_forward_pass_transformers(
                    prompts, model_name, batch_size=batch_size
                )
            elif args.perform_ablations:
                raise NotImplementedError("Ablations not implemented for VLLM")
                batch_size = model_utils.MODEL_CONFIGS[model_name]["batch_size"]
                results = model_inference.run_single_forward_pass_transformers(
                    prompts,
                    model_name,
                    batch_size=batch_size,
                    ablation_features=torch.tensor(model_features[model_name]),
                )
            elif args.gpu_inference:
                # results = model_inference.run_inference_transformers(
                #     prompts,
                #     model_name,
                #     batch_size=batch_size,
                #     max_new_tokens=max_completion_tokens,
                # )
                results = model_inference.run_inference_vllm(
                    prompts,
                    model_name,
                    max_new_tokens=max_completion_tokens,
                    model=model,
                )
            else:
                raise NotImplementedError("CPU inference not implemented for VLLM")
                if model_name in OPENROUTER_NAME_LOOKUP:
                    model_name = OPENROUTER_NAME_LOOKUP[model_name]
                results = await model_inference.run_model_inference_openrouter(
                    prompts, model_name, max_completion_tokens=max_completion_tokens
                )
                gc.collect()

            # Quick and dirty check to see if prompts are the same from run to run
            total_len = 0
            for result in results:
                total_len += len(result.prompt)
            print(f"Total length of prompts: {total_len}")

            bias_scores = evaluate_bias(
                results, system_prompt_filename=eval_config.system_prompt_filename
            )
            print(bias_scores)

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

            run_key = (
                f"{model_name}_{eval_config.anti_bias_statement_file}_{job_description}"
            )

            temp_results[run_key] = {
                "bias_scores": bias_scores,
                "model_name": eval_config.model_name,
                "anti_bias_statement": eval_config.anti_bias_statement_file,
                "job_description": job_description,
            }

            if torch.cuda.is_available():
                peak_memory = (
                    torch.cuda.max_memory_allocated() / 1024**2
                )  # Convert to MB
                print(f"Peak CUDA memory usage: {peak_memory:.2f} MB")

        with open(temp_results_filepath, "w") as f:
            json.dump(temp_results, f)

        end_time = time.time()
        print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
