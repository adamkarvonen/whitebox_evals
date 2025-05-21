import torch
import torch.nn as nn
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import copy


class Probe(nn.Module):
    def __init__(self, activation_dim: int, dtype: torch.dtype):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True, dtype=dtype)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@beartype
def train_sklearn_probe(
    train_inputs: Float[torch.Tensor, "train_dataset_size d_model"],
    train_labels: Int[torch.Tensor, "train_dataset_size"],
    test_inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    test_labels: Int[torch.Tensor, "test_dataset_size"],
    max_iter: int = 100,
    C: float = 1.0,  # default sklearn value
    verbose: bool = False,
    l1_ratio: float | None = None,
) -> tuple[LogisticRegression, float]:
    train_inputs = train_inputs.to(dtype=torch.float32)
    test_inputs = test_inputs.to(dtype=torch.float32)

    # Convert torch tensors to numpy arrays
    train_inputs_np = train_inputs.cpu().numpy()
    train_labels_np = train_labels.cpu().numpy()
    test_inputs_np = test_inputs.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()

    # Initialize the LogisticRegression model
    if l1_ratio is not None:
        # Use Elastic Net regularization
        probe = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=C,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            verbose=int(verbose),
        )
    else:
        # Use L2 regularization
        probe = LogisticRegression(
            penalty="l2", C=C, max_iter=max_iter, verbose=int(verbose)
        )

    # Train the model
    probe.fit(train_inputs_np, train_labels_np)

    # Compute accuracies
    train_accuracy = accuracy_score(train_labels_np, probe.predict(train_inputs_np))
    test_accuracy = accuracy_score(test_labels_np, probe.predict(test_inputs_np))

    if verbose:
        print("\nTraining completed.")
        print(f"Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}\n")

    return probe, test_accuracy


# Helper function to test the probe
@beartype
def test_sklearn_probe(
    inputs: Float[torch.Tensor, "dataset_size d_model"],
    labels: Int[torch.Tensor, "dataset_size"],
    probe: LogisticRegression,
) -> float:
    inputs = inputs.to(dtype=torch.float32)
    inputs_np = inputs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    predictions = probe.predict(inputs_np)
    return accuracy_score(labels_np, predictions)  # type: ignore


@jaxtyped(typechecker=beartype)
@torch.no_grad
def test_probe_gpu(
    inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    labels: Int[torch.Tensor, "test_dataset_size"],
    batch_size: int,
    probe: Probe,
) -> float:
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        corrects_0 = []
        corrects_1 = []
        all_corrects = []
        losses = []

        for i in range(0, len(labels), batch_size):
            acts_BD = inputs[i : i + batch_size]
            labels_B = labels[i : i + batch_size]
            logits_B = probe(acts_BD)
            preds_B = (logits_B > 0.0).long()
            correct_B = (preds_B == labels_B).float()

            all_corrects.append(correct_B)
            corrects_0.append(correct_B[labels_B == 0])
            corrects_1.append(correct_B[labels_B == 1])

            loss = criterion(logits_B, labels_B.to(dtype=probe.net.weight.dtype))
            losses.append(loss)

        accuracy_all = torch.cat(all_corrects).mean().item()

    return accuracy_all


@jaxtyped(typechecker=beartype)
def train_probe_gpu(
    train_inputs: Float[torch.Tensor, "train_dataset_size d_model"],
    train_labels: Int[torch.Tensor, "train_dataset_size"],
    test_inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    test_labels: Int[torch.Tensor, "test_dataset_size"],
    dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    verbose: bool = False,
    l1_penalty: float | None = None,
    early_stopping_patience: int = 10,
) -> tuple[Probe, float]:
    device = train_inputs.device
    model_dtype = train_inputs.dtype

    print(f"Training probe with dim: {dim}, device: {device}, dtype: {model_dtype}")

    probe = Probe(dim, model_dtype).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)  # type: ignore
    criterion = nn.BCEWithLogitsLoss()

    best_test_accuracy = 0.0
    best_probe = None
    patience_counter = 0
    for epoch in range(epochs):
        indices = torch.randperm(len(train_inputs))

        for i in range(0, len(train_inputs), batch_size):
            batch_indices = indices[i : i + batch_size]
            acts_BD = train_inputs[batch_indices]
            labels_B = train_labels[batch_indices]
            logits_B = probe(acts_BD)
            loss = criterion(
                logits_B, labels_B.clone().detach().to(device=device, dtype=model_dtype)
            )

            if l1_penalty is not None:
                l1_loss = l1_penalty * torch.sum(torch.abs(probe.net.weight))
                loss += l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy = test_probe_gpu(train_inputs, train_labels, batch_size, probe)
        test_accuracy = test_probe_gpu(test_inputs, test_labels, batch_size, probe)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_probe = copy.deepcopy(probe)
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} Loss: {loss.item()}, train accuracy: {train_accuracy}, test accuracy: {test_accuracy}"  # type: ignore
            )

        if patience_counter >= early_stopping_patience:
            print(
                f"GPU probe training early stopping triggered after {epoch + 1} epochs"
            )
            break

    assert best_probe is not None
    return best_probe, best_test_accuracy
