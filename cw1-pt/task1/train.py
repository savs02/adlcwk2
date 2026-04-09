"""
GenAI usage statement:
- Tool used: Claude
- Assistance received: documentation
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from common import (
    BASELINE_MODEL_PATH,
    REGULARIZED_MODEL_PATH,
    ensure_directories,
    evaluate_model,
    load_datasets,
    make_loaders,
    save_history,
    save_model,
    set_seed,
    train_model,
    build_model,
)


SEED = 7
EPOCHS = 25
BATCH_SIZE = 128
VALIDATION_FRACTION = 0.1
DEVICE = torch.device("cpu")


def print_history(name: str, history: Dict[str, list[float]]) -> None:
    """Print concise epoch summaries.

    Args:
        name: str, experiment label.
        history: dict[str, list[float]], scalar metrics by epoch.
    """

    print(f"\n{name}")
    print("epoch | train_loss | train_acc | val_loss | val_acc")
    for epoch_index, _ in enumerate(history["train_loss"], start=1):
        print(
            f"{epoch_index:>5d} | "
            f"{history['train_loss'][epoch_index - 1]:>10.4f} | "
            f"{history['train_accuracy'][epoch_index - 1]:>9.4f} | "
            f"{history['val_loss'][epoch_index - 1]:>8.4f} | "
            f"{history['val_accuracy'][epoch_index - 1]:>7.4f}"
        )


def main() -> None:
    """Train the baseline and regularized models for Task 1."""

    ensure_directories()
    set_seed(SEED)

    train_dataset, val_dataset, test_dataset = load_datasets(
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
    )
    train_loader, val_loader, test_loader = make_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=BATCH_SIZE,
    )

    criterion = nn.CrossEntropyLoss()

    baseline_config = {
        "name": "baseline",
        "hidden_dims": [1024, 512, 512, 256, 128, 64],
        "dropout_rate": 0.0,
        "weight_decay": 0.0,
        "learning_rate": 0.08,
        "momentum": 0.9,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_fraction": VALIDATION_FRACTION,
        "seed": SEED,
        "optimizer": "SGD",
        "dataset": "FashionMNIST",
    }
    regularized_config = {
        "name": "regularized",
        "hidden_dims": [1024, 512, 512, 256, 128, 64],
        "dropout_rate": 0.15,
        "weight_decay": 1e-4,
        "learning_rate": 0.05,
        "momentum": 0.9,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_fraction": VALIDATION_FRACTION,
        "seed": SEED,
        "optimizer": "SGD",
        "dataset": "FashionMNIST",
    }

    baseline_model = build_model(baseline_config)
    baseline_optimizer = torch.optim.SGD(
        baseline_model.parameters(),
        lr=float(baseline_config["learning_rate"]),
        momentum=float(baseline_config["momentum"]),
        weight_decay=float(baseline_config["weight_decay"]),
    )
    baseline_history = train_model(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=baseline_optimizer,
        criterion=criterion,
        epochs=EPOCHS,
        device=DEVICE,
    )
    save_model(baseline_model, baseline_config, BASELINE_MODEL_PATH)

    regularized_model = build_model(regularized_config)
    regularized_optimizer = torch.optim.SGD(
        regularized_model.parameters(),
        lr=float(regularized_config["learning_rate"]),
        momentum=float(regularized_config["momentum"]),
        weight_decay=float(regularized_config["weight_decay"]),
    )
    regularized_history = train_model(
        model=regularized_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=regularized_optimizer,
        criterion=criterion,
        epochs=EPOCHS,
        device=DEVICE,
    )
    save_model(regularized_model, regularized_config, REGULARIZED_MODEL_PATH)

    save_history(
        {
            "baseline": baseline_history,
            "regularized": regularized_history,
        }
    )

    baseline_test_loss, baseline_test_accuracy = evaluate_model(
        baseline_model, test_loader, criterion, DEVICE
    )
    regularized_test_loss, regularized_test_accuracy = evaluate_model(
        regularized_model, test_loader, criterion, DEVICE
    )

    print_history("Baseline model", baseline_history.__dict__)
    print_history("Regularized model", regularized_history.__dict__)

    print("\nHeld-out test summary")
    print(
        f"Baseline:     loss={baseline_test_loss:.4f}, accuracy={baseline_test_accuracy:.4f}"
    )
    print(
        f"Regularized:  loss={regularized_test_loss:.4f}, accuracy={regularized_test_accuracy:.4f}"
    )
    print(f"\nSaved baseline checkpoint to {BASELINE_MODEL_PATH}")
    print(f"Saved regularized checkpoint to {REGULARIZED_MODEL_PATH}")


if __name__ == "__main__":
    main()
