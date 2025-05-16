# %%
import torch
import pickle
import matplotlib.pyplot as plt

# %%

initial_filename = "initial_mse_bl.pkl"
final_filename = "mse_bl_epoch_0.pkl"

with open(initial_filename, "rb") as f:
    initial_mse_bl = pickle.load(f)

with open(final_filename, "rb") as f:
    final_mse_bl = pickle.load(f)


# %%

print(initial_mse_bl[0].shape)
print(initial_mse_bl[1].shape)
# %%
# visualise_mse.py
import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_pickle(path: str | Path) -> List[torch.Tensor]:
    with open(path, "rb") as f:
        return pickle.load(f)


def flatten_values(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate every value across batches and sequence positions."""
    return torch.cat([t.float().cpu().flatten() for t in tensor_list])


def plot_histogram(values: torch.Tensor, title: str, fname: str) -> None:
    plt.figure()
    plt.hist(values.cpu().numpy(), bins=100)
    plt.title(title)
    plt.xlabel("MSE value")
    plt.ylabel("Count")
    plt.yscale("log")
    plt.tight_layout()
    # plt.savefig(fname, dpi=150)
    plt.show()
    plt.close()


def plot_tokenwise_mean(
    tensor_list: List[torch.Tensor], title: str, fname: str
) -> None:
    """
    Compute the mean MSE for each token position across *all* examples that
    actually contain that position.
    """
    # Accumulate sums and counts per position
    max_len = max(t.shape[1] for t in tensor_list)
    sums = torch.zeros(max_len)
    counts = torch.zeros(max_len)

    for t in tensor_list:
        # mean across batch dimension to get [seq_len] vector
        seq_means = t.float().cpu().mean(dim=0)
        L = seq_means.shape[0]
        sums[:L] += seq_means
        counts[:L] += 1

    mean_per_pos = (
        sums / counts
    )  # counts will never be zero for the first max_len tokens
    positions = np.arange(1, max_len + 1)

    plt.figure()
    plt.plot(positions, mean_per_pos.cpu().numpy())
    plt.title(title)
    plt.xlabel("Token position")
    plt.ylabel("Mean MSE")
    plt.yscale("log")
    plt.tight_layout()
    # plt.savefig(fname, dpi=150)
    plt.show()
    plt.close()


def main() -> None:
    initial = load_pickle("initial_mse_bl.pkl")
    final = load_pickle("mse_bl_epoch_0.pkl")

    # 1. Histograms
    plot_histogram(
        flatten_values(initial),
        "Initial MSE — Histogram",
        "initial_mse_hist.png",
    )
    plot_histogram(
        flatten_values(final),
        "Final MSE — Histogram",
        "final_mse_hist.png",
    )

    # 2. Mean-per-token plots
    plot_tokenwise_mean(
        initial,
        "Initial MSE — Mean per token position",
        "initial_mse_token_mean.png",
    )
    plot_tokenwise_mean(
        final,
        "Final MSE — Mean per token position",
        "final_mse_token_mean.png",
    )

    print("Saved four figures:")
    for f in Path(".").glob("*_mse_*.png"):
        print(f" • {f}")


if __name__ == "__main__":
    main()

# %%
