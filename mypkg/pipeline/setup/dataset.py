import pandas as pd
import random
from datasets import load_dataset

from mypkg.pipeline.setup.config import load_config


def load_raw_dataset():
    """
    Loads the preprocessed CSV from the path specified in the config.
    Returns a Pandas DataFrame with all resumes and metadata.
    """
    config = load_config()
    dataset_path = config["dataset"]["path"]
    df = pd.read_csv(dataset_path)
    return df


def filter_by_industry(df, industry):
    """
    Filters the DataFrame to include only rows for the specified industry:
    'INFORMATION-TECHNOLOGY', 'CONSTRUCTION', or 'TEACHER'.
    """
    return df[df["Category"] == industry].reset_index(drop=True)


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


def filter_by_demographics(df, gender=None, race=None, politics=None):
    """
    Further filters the DataFrame by demographic attributes.
    If any of these parameters are None, that attribute is ignored.
    """
    if gender is not None:
        df = df[df["Gender"] == gender]
    if race is not None:
        df = df[df["Race"] == race]
    if politics is not None:
        df = df[df["Political_orientation"] == politics]
    return df.reset_index(drop=True)


def load_full_anthropic_dataset(downsample_questions: int = 150):
    explicit_dataset = load_dataset("Anthropic/discrim-eval", "explicit")
    implicit_dataset = load_dataset("Anthropic/discrim-eval", "implicit")

    explicit_df = pd.DataFrame(explicit_dataset["train"])
    implicit_df = pd.DataFrame(implicit_dataset["train"])

    df = pd.concat([explicit_df, implicit_df])

    df["race"] = df["race"].str.lower()
    df["gender"] = df["gender"].str.lower()

    retain_races = ["white", "black"]
    retain_ages = [20, 30, 40]
    retain_questions = list(range(0, downsample_questions))
    retain_genders = ["male", "female"]

    print(f"DF length before filtering: {len(df)}")

    df = df[df["race"].isin(retain_races)]
    df = df[df["age"].isin(retain_ages)]
    df = df[df["decision_question_id"].isin(retain_questions)]
    df = df[df["gender"].isin(retain_genders)]

    df = df.rename(
        columns={
            "race": "Race",
            "gender": "Gender",
        }
    )

    print(f"DF length after filtering: {len(df)}")

    return df


def summarize_resume(full_text, max_chars=1024):
    """
    Summarize the resume if needed.
    """
    raise ValueError("Summarizing resumes is not supported yet")
    return (full_text[:max_chars] + "...") if len(full_text) > max_chars else full_text


def prepare_dataset_for_model(df, mode="full"):
    """
    Preprocesses the DataFrame depending on whether you want a 'full' resume
    or a 'summary' version for smaller models.
    """
    if mode == "summary":
        df["Resume"] = df["Resume"].apply(lambda text: summarize_resume(str(text)))
    return df
