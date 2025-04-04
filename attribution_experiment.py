import torch
import einops
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import os
import random
import json
import pickle
import argparse
import gc
import mypkg.whitebox_infra.attribution as attribution
import mypkg.whitebox_infra.dictionaries.batch_topk_sae as batch_topk_sae
import mypkg.whitebox_infra.data_utils as data_utils
import mypkg.whitebox_infra.model_utils as model_utils
import mypkg.whitebox_infra.interp_utils as interp_utils
import mypkg.pipeline.setup.dataset as dataset_setup
import mypkg.pipeline.infra.hiring_bias_prompts as hiring_bias_prompts
from mypkg.eval_config import EvalConfig


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model_name
    # model_name = "google/gemma-2-9b-it"
    # model_name = "google/gemma-2-27b-it"
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name == "google/gemma-2-27b-it":
        gradient_checkpointing = True
        batch_size = 1
    elif model_name == "mistralai/Mistral-Small-24B-Instruct-2501":
        gradient_checkpointing = False
        batch_size = 2
    else:
        gradient_checkpointing = False
        batch_size = 4

    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    chosen_layer_percentage = [25]

    trainer_id = 1

    chosen_layers = []
    for layer_percent in chosen_layer_percentage:
        chosen_layers.append(
            model_utils.MODEL_CONFIGS[model_name]["layer_mappings"][layer_percent][
                "layer"
            ]
        )

    sae = model_utils.load_model_sae(
        model_name, device, dtype, chosen_layer_percentage[0], trainer_id=trainer_id
    )

    submodules = [model_utils.get_submodule(model, chosen_layers[0])]

    eval_config = EvalConfig(
        model_name=model_name,
        political_orientation=True,
        pregnancy=False,
        employment_gap=False,
        anthropic_dataset=False,
        downsample=args.downsample,
        gpu_inference=True,
        anti_bias_statement_file=args.anti_bias_statement_file,
        job_description_file="short_meta_job_description.txt",
        system_prompt_filename="yes_no.txt",
    )

    df = dataset_setup.load_raw_dataset()

    industry = "INFORMATION-TECHNOLOGY"
    downsample = eval_config.downsample
    random_seed = eval_config.random_seed

    random.seed(random_seed)
    torch.manual_seed(random_seed)

    df = dataset_setup.filter_by_industry(df, industry)

    df = dataset_setup.balanced_downsample(df, downsample, random_seed)

    bias_categories_to_test = ["political_orientation", "gender", "race"]

    output_dir = os.path.join(args.output_dir, model_name.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(
        output_dir,
        f"{args.anti_bias_statement_file.replace('.txt', '')}_trainer_{trainer_id}_attrib_data.pt",
    )
    all_data = {}

    # --- Loop over the bias categories ---
    for bias_category in bias_categories_to_test:
        args = hiring_bias_prompts.HiringBiasArgs(
            political_orientation=bias_category == "political_orientation",
            employment_gap=bias_category == "employment_gap",
            pregnancy=bias_category == "pregnancy",
            race=bias_category == "race",
            gender=bias_category == "gender",
        )

        prompts = hiring_bias_prompts.create_all_prompts_hiring_bias(
            df, args, eval_config
        )

        train_texts, train_labels = (
            hiring_bias_prompts.process_hiring_bias_resumes_prompts(prompts, args)
        )

        train_texts = model_utils.add_chat_template(train_texts, model_name)

        dataloader = data_utils.create_simple_dataloader(
            train_texts, train_labels, model_name, device, batch_size=batch_size
        )

        # Build the custom loss function
        yes_vs_no_loss_fn = attribution.make_yes_no_loss_fn(
            tokenizer,
            yes_candidates=["yes", " yes", "Yes", " Yes", "YES", " YES"],
            no_candidates=["no", " no", "No", " No", "NO", " NO"],
            device=device,
        )

        gc.collect()
        torch.cuda.empty_cache()

        effects_F, error_effect = attribution.get_effects(
            model,
            sae,
            dataloader,
            yes_vs_no_loss_fn,
            submodules,
            chosen_layers,
            device,
        )

        # Print peak memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
            print(f"Peak CUDA memory usage: {peak_memory:.2f} MB")

        top_k_ids = effects_F.abs().topk(20).indices
        print(top_k_ids)

        top_k_vals = effects_F[top_k_ids]
        print(top_k_vals)

        print(error_effect)

        all_data[bias_category] = {
            "effects_F": effects_F.cpu(),
            "error_effect": error_effect.cpu(),
        }
        torch.save(all_data, output_filename)

    torch.save(all_data, output_filename)
    print(f"Experiment finished. Results saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run attribution experiment with specified model and anti-bias statements."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Ministral-8B-Instruct-2410",
        help="Name of the Hugging Face model to use.",
    )
    parser.add_argument(
        "--anti_bias_statement_file",
        type=str,
        default="v0.txt",
        help="Path to the anti-bias statement file.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=100,
        help="Downsample the dataset to this number of samples.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="attribution_results",
        help="Output directory for the attribution results.",
    )

    args = parser.parse_args()
    main(args)
