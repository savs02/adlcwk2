"""
GenAI usage statement:
- Tool used: Claude
- Assistance received: documentation
"""

from __future__ import annotations

import json
import random
import ssl
import urllib.request
from dataclasses import dataclass, asdict
from gzip import open as gzip_open
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
DATA_DIR = ROOT / "data" / "fashion-mnist"
HISTORY_PATH = ARTIFACTS / "training_history.json"
BASELINE_MODEL_PATH = MODELS / "baseline_model.pth"
REGULARIZED_MODEL_PATH = MODELS / "regularized_model.pth"
PLOT_PATH = ROOT / "generalization_gap.png"

IMAGE_SIZE = 28 * 28
NUM_CLASSES = 10
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
FASHION_MNIST_URLS = {
    "train_images": [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz",
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
    ],
    "train_labels": [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz",
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
    ],
    "test_images": [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz",
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
    ],
    "test_labels": [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz",
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
    ],
}


@dataclass
class History:
    """Container for epoch-wise scalar metrics.

    Attributes:
        train_loss: list[float], training loss per epoch.
        train_accuracy: list[float], training accuracy per epoch.
        val_loss: list[float], validation loss per epoch.
        val_accuracy: list[float], validation accuracy per epoch.
    """

    train_loss: List[float]
    train_accuracy: List[float]
    val_loss: List[float]
    val_accuracy: List[float]


def ensure_directories() -> None:
    """Create output directories used by the task."""

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Set random seeds for deterministic CPU training.

    Args:
        seed: int, seed value used for Python and PyTorch RNGs.
    """

    random.seed(seed)
    torch.manual_seed(seed)


class DeepFashionMLP(nn.Module):
    """Fully connected classifier built only from primitive layers.

    Args:
        hidden_dims: list[int], hidden layer widths.
        dropout_rate: float, dropout probability applied after each hidden block.

    Returns:
        torch.Tensor of shape [batch_size, NUM_CLASSES].
    """

    def __init__(self, hidden_dims: List[int], dropout_rate: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        input_dim = IMAGE_SIZE
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, NUM_CLASSES))
        self.network = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            images: torch.Tensor, shape [batch_size, 1, 28, 28].

        Returns:
            torch.Tensor, logits with shape [batch_size, NUM_CLASSES].
        """

        flat = images.view(images.size(0), -1)
        return self.network(flat)


class FashionMNISTTensorDataset(Dataset):
    """FashionMNIST dataset backed by preloaded tensors.

    Args:
        images: torch.Tensor, shape [N, 1, 28, 28] with values in [0, 1].
        labels: torch.Tensor, shape [N] containing class indices.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        """Return dataset size."""

        return int(self.labels.size(0))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Return one image-label pair.

        Args:
            index: int, sample index.

        Returns:
            tuple(torch.Tensor, int), image tensor and class label.
        """

        return self.images[index], int(self.labels[index].item())


def build_model(config: Dict[str, object]) -> DeepFashionMLP:
    """Instantiate the coursework model from a config mapping.

    Args:
        config: dict, expects keys `hidden_dims` and `dropout_rate`.

    Returns:
        DeepFashionMLP configured from the supplied values.
    """

    return DeepFashionMLP(
        hidden_dims=list(config["hidden_dims"]),
        dropout_rate=float(config["dropout_rate"]),
    )


def download_file(urls: List[str], destination: Path) -> None:
    """Download a single FashionMNIST archive if not already present.

    Args:
        urls: list[str], candidate remote archive URLs.
        destination: Path, local `.gz` destination.
    """

    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in urls:
        try:
            with urllib.request.urlopen(url) as response:
                destination.write_bytes(response.read())
            return
        except Exception as error:
            last_error = error
        try:
            insecure_context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=insecure_context) as response:
                destination.write_bytes(response.read())
            return
        except Exception as error:
            last_error = error
            if destination.exists():
                destination.unlink()
    if last_error is not None:
        raise RuntimeError(f"Failed to download {destination.name}") from last_error


def ensure_fashion_mnist_downloaded() -> None:
    """Download all FashionMNIST IDX gzip files required for the task."""

    for _, urls in FASHION_MNIST_URLS.items():
        download_file(urls, DATA_DIR / Path(urls[0]).name)


def read_idx_images(path: Path) -> torch.Tensor:
    """Read FashionMNIST image tensors from an IDX gzip archive.

    Args:
        path: Path, gzip-compressed IDX image file.

    Returns:
        torch.Tensor with shape [N, 1, 28, 28] and dtype float32.
    """

    with gzip_open(path, "rb") as file_obj:
        raw = file_obj.read()
    data = torch.frombuffer(bytearray(raw[16:]), dtype=torch.uint8).clone()
    images = data.view(-1, 1, 28, 28).float() / 255.0
    return images


def read_idx_labels(path: Path) -> torch.Tensor:
    """Read FashionMNIST labels from an IDX gzip archive.

    Args:
        path: Path, gzip-compressed IDX label file.

    Returns:
        torch.Tensor with shape [N] and dtype long.
    """

    with gzip_open(path, "rb") as file_obj:
        raw = file_obj.read()
    return torch.frombuffer(bytearray(raw[8:]), dtype=torch.uint8).clone().long()


def load_datasets(validation_fraction: float, seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """Download FashionMNIST and create train/validation/test splits.

    Args:
        validation_fraction: float, fraction of the training set held out for validation.
        seed: int, seed used for deterministic splitting.

    Returns:
        tuple of train, validation, and test datasets.
    """

    ensure_fashion_mnist_downloaded()
    train_images = read_idx_images(DATA_DIR / "train-images-idx3-ubyte.gz")
    train_labels = read_idx_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")
    test_images = read_idx_images(DATA_DIR / "t10k-images-idx3-ubyte.gz")
    test_labels = read_idx_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz")

    full_train = FashionMNISTTensorDataset(train_images, train_labels)
    test_dataset = FashionMNISTTensorDataset(test_images, test_labels)

    validation_size = int(len(full_train) * validation_fraction)
    train_size = len(full_train) - validation_size
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_dataset, val_dataset, test_dataset


def make_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for each split.

    Args:
        train_dataset: Dataset, training subset.
        val_dataset: Dataset, validation subset.
        test_dataset: Dataset, held-out test set.
        batch_size: int, minibatch size.

    Returns:
        tuple of train, validation, and test dataloaders.
    """

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one epoch for either training or evaluation.

    Args:
        model: nn.Module, network under evaluation.
        loader: DataLoader, minibatch iterator.
        criterion: nn.Module, scalar loss function.
        optimizer: Optimizer or None, update rule for training mode.
        device: torch.device, compute device.

    Returns:
        tuple(float, float), mean loss and accuracy for the epoch.
    """

    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate a model without gradient tracking.

    Args:
        model: nn.Module, trained model.
        loader: DataLoader, evaluation data.
        criterion: nn.Module, loss function.
        device: torch.device, compute device.

    Returns:
        tuple(float, float), mean loss and accuracy.
    """

    with torch.no_grad():
        return run_epoch(model, loader, criterion, optimizer=None, device=device)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
) -> History:
    """Train a model and record epoch metrics.

    Args:
        model: nn.Module, network to fit.
        train_loader: DataLoader, training set.
        val_loader: DataLoader, validation set.
        optimizer: Optimizer, gradient-based update rule.
        criterion: nn.Module, scalar loss function.
        epochs: int, number of complete passes through training data.
        device: torch.device, compute device.

    Returns:
        History containing training and validation metrics.
    """

    train_loss: List[float] = []
    train_accuracy: List[float] = []
    val_loss: List[float] = []
    val_accuracy: List[float] = []

    model.to(device)
    print("epoch | train_loss | train_acc | val_loss | val_acc", flush=True)
    for epoch in range(1, epochs + 1):
        epoch_train_loss, epoch_train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        with torch.no_grad():
            epoch_val_loss, epoch_val_accuracy = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
            )

        train_loss.append(epoch_train_loss)
        train_accuracy.append(epoch_train_accuracy)
        val_loss.append(epoch_val_loss)
        val_accuracy.append(epoch_val_accuracy)

        print(
            f"{epoch:>5d} | {epoch_train_loss:>10.4f} | {epoch_train_accuracy:>9.4f} |"
            f" {epoch_val_loss:>8.4f} | {epoch_val_accuracy:>7.4f}",
            flush=True,
        )

    return History(
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
    )


def save_model(model: nn.Module, config: Dict[str, object], path: Path) -> None:
    """Save a model checkpoint.

    Args:
        model: nn.Module, trained network.
        config: dict, model hyperparameters and metadata.
        path: Path, checkpoint output location.
    """

    torch.save({"model_state": model.state_dict(), "config": config}, path)


def load_model(path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, object]]:
    """Load a saved model checkpoint.

    Args:
        path: Path, checkpoint file.
        device: torch.device, target device.

    Returns:
        tuple(nn.Module, dict), restored model and stored config.
    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint["config"]


def save_history(histories: Dict[str, History]) -> None:
    """Write all training histories to JSON.

    Args:
        histories: dict[str, History], named metric containers.
    """

    payload = {name: asdict(history) for name, history in histories.items()}
    HISTORY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_history() -> Dict[str, History]:
    """Read training histories from JSON.

    Returns:
        dict[str, History], parsed metric containers.
    """

    payload = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    return {name: History(**history) for name, history in payload.items()}


