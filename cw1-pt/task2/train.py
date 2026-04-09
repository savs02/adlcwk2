"""
GenAI usage statement:
- Tool used: Claude
- Assistance received: documentation
"""

from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, Subset

from common import (
    HISTORY_PATH,
    MODEL_PATH,
    History,
    build_model,
    clean_accuracy_epoch,
    ensure_directories,
    evaluate_epoch,
    load_datasets,
    make_loaders,
    save_history,
    save_model,
    set_seed,
    train_epoch,
)


SEED = 13
DEVICE = torch.device("cpu")
BATCH_SIZE = 128
TRAIN_METRIC_SAMPLES = 2048
VALIDATION_FRACTION = 0.1
MAX_EPOCHS = 20
PATIENCE = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.4
SMOOTHING = 0.1


def main() -> None:
    """Train Task 2 model with MixUp, label smoothing, and early stopping."""

    ensure_directories()
    set_seed(SEED)

    train_dataset, val_dataset, test_dataset = load_datasets(
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
    )
    metric_count = min(TRAIN_METRIC_SAMPLES, len(train_dataset))
    train_metric_dataset = Subset(train_dataset, list(range(metric_count)))
    train_loader, val_loader, _ = make_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=BATCH_SIZE,
    )
    train_metric_loader = DataLoader(train_metric_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    best_val_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = copy.deepcopy(model.state_dict())

    print("Training Task 2 model with custom MixUp and label smoothing", flush=True)
    print("epoch | train_loss | clean_train_acc | val_loss | val_acc | status", flush=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_train_loss, _ = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            smoothing=SMOOTHING,
            mixup_alpha=MIXUP_ALPHA,
            device=DEVICE,
        )
        epoch_val_loss, epoch_val_accuracy = evaluate_epoch(
            model=model,
            loader=val_loader,
            smoothing=SMOOTHING,
            device=DEVICE,
        )
        epoch_clean_train_accuracy = clean_accuracy_epoch(
            model=model,
            loader=train_metric_loader,
            device=DEVICE,
        )

        train_loss.append(epoch_train_loss)
        train_accuracy.append(epoch_clean_train_accuracy)
        val_loss.append(epoch_val_loss)
        val_accuracy.append(epoch_val_accuracy)

        status = "keep"
        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
            status = "best"
        else:
            epochs_without_improvement += 1
            status = f"wait {epochs_without_improvement}/{PATIENCE}"

        print(
            f"{epoch:>5d} | {epoch_train_loss:>10.4f} | {epoch_clean_train_accuracy:>15.4f} | "
            f"{epoch_val_loss:>8.4f} | {epoch_val_accuracy:>7.4f} | {status}",
            flush=True,
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.", flush=True)
            break

    model.load_state_dict(best_state)

    history = History(
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
        best_epoch=best_epoch,
    )
    save_history(history)

    config = {
        "dataset": "FashionMNIST",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "mixup_alpha": MIXUP_ALPHA,
        "smoothing": SMOOTHING,
        "batch_size": BATCH_SIZE,
        "train_metric_samples": metric_count,
        "validation_fraction": VALIDATION_FRACTION,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "seed": SEED,
    }
    save_model(model, config, best_epoch)

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}", flush=True)
    print(f"Saved model to {MODEL_PATH}", flush=True)
    print(f"Saved history to {HISTORY_PATH}", flush=True)


if __name__ == "__main__":
    main()
