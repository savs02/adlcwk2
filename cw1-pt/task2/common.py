"""
GenAI usage statement:
- Tool used: OpenAI Codex.
- Assistance received: scaffolding, implementation drafting, and code review.
- Human verification: the final code structure, hyperparameters, and task-specific logic
  were checked and adjusted to satisfy the coursework specification.
"""

from __future__ import annotations

import json
import random
import ssl
import urllib.request
from dataclasses import asdict, dataclass
from gzip import open as gzip_open
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
LOCAL_DATA_DIR = ROOT / "data" / "fashion-mnist"
SHARED_TASK1_DATA_DIR = ROOT.parent / "task1" / "data" / "fashion-mnist"
MODEL_PATH = MODELS / "mixup_label_smoothing_model.pth"
HISTORY_PATH = ARTIFACTS / "training_history.json"
ROBUSTNESS_PLOT_PATH = ROOT / "robustness_demo.png"

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
    """Container for Task 2 metrics.

    Attributes:
        train_loss: list[float], epoch training loss.
        train_accuracy: list[float], epoch training accuracy on unmixed labels.
        val_loss: list[float], epoch validation loss.
        val_accuracy: list[float], epoch validation accuracy.
        best_epoch: int, epoch index selected by early stopping.
    """

    train_loss: List[float]
    train_accuracy: List[float]
    val_loss: List[float]
    val_accuracy: List[float]
    best_epoch: int


def ensure_directories() -> None:
    """Create Task 2 output directories."""

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Set Python and PyTorch RNG seeds.

    Args:
        seed: int, seed value.
    """

    random.seed(seed)
    torch.manual_seed(seed)


class FashionMNISTTensorDataset(Dataset):
    """FashionMNIST dataset backed by tensors.

    Args:
        images: torch.Tensor, shape [N, 1, 28, 28].
        labels: torch.Tensor, shape [N].
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.size(0))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.images[index], int(self.labels[index].item())


class FashionCNN(nn.Module):
    """Small convolutional classifier built from primitive layers.

    Architecture (all layers from nn primitives, no pre-defined blocks):
      features:
        Conv2d(1→32, 3×3, padding=1)  → ReLU
        Conv2d(32→64, 3×3, padding=1) → ReLU → MaxPool2d(2) → Dropout(0.25)
        Conv2d(64→128, 3×3, padding=1)→ ReLU → MaxPool2d(2)
        Output: [batch, 128, 7, 7] → flattened to 6272
      classifier:
        Linear(6272→256) → ReLU → Dropout(0.5) → Linear(256→10)

    Returns:
        torch.Tensor, class logits of shape [batch_size, NUM_CLASSES].
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 0  [1,28,28]→[32,28,28]
            nn.ReLU(),                                      # 1
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 2  →[64,28,28]
            nn.ReLU(),                                      # 3
            nn.MaxPool2d(2),                                # 4  →[64,14,14]
            nn.Dropout(0.25),                               # 5
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 6  →[128,14,14]
            nn.ReLU(),                                      # 7
            nn.MaxPool2d(2),                                # 8  →[128,7,7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                   # 0  6272
            nn.Linear(6272, 256),          # 1
            nn.ReLU(),                      # 2
            nn.Dropout(0.5),               # 3
            nn.Linear(256, NUM_CLASSES),   # 4
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the convolutional feature extractor and classifier head.

        Args:
            images: torch.Tensor, shape [batch_size, 1, 28, 28].

        Returns:
            torch.Tensor, logits of shape [batch_size, NUM_CLASSES].
        """

        return self.classifier(self.features(images))


def build_model() -> FashionCNN:
    """Instantiate the Task 2 convolutional classifier."""

    return FashionCNN()


def candidate_data_directories() -> List[Path]:
    """Return possible directories containing FashionMNIST archives."""

    return [LOCAL_DATA_DIR, SHARED_TASK1_DATA_DIR]


def resolve_data_directory() -> Path:
    """Select an existing data directory or the local download target."""

    required = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for directory in candidate_data_directories():
        if all((directory / name).exists() for name in required):
            return directory
    return LOCAL_DATA_DIR


def download_file(urls: List[str], destination: Path) -> None:
    """Download an archive, trying secure and fallback SSL contexts.

    Args:
        urls: list[str], candidate URLs for one archive.
        destination: Path, local output file.
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
        except Exception as error:  # noqa: BLE001
            last_error = error
        try:
            insecure_context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=insecure_context) as response:
                destination.write_bytes(response.read())
            return
        except Exception as error:  # noqa: BLE001
            last_error = error
            if destination.exists():
                destination.unlink()
    if last_error is not None:
        raise RuntimeError(f"Failed to download {destination.name}") from last_error


def ensure_fashion_mnist_downloaded() -> Path:
    """Ensure FashionMNIST archives exist and return the chosen data directory."""

    resolved = resolve_data_directory()
    if resolved != LOCAL_DATA_DIR:
        return resolved
    for urls in FASHION_MNIST_URLS.values():
        download_file(urls, LOCAL_DATA_DIR / Path(urls[0]).name)
    return LOCAL_DATA_DIR


def read_idx_images(path: Path) -> torch.Tensor:
    """Read image tensors from an IDX gzip archive.

    Args:
        path: Path, compressed image archive.

    Returns:
        torch.Tensor, shape [N, 1, 28, 28].
    """

    with gzip_open(path, "rb") as file_obj:
        raw = file_obj.read()
    data = torch.frombuffer(bytearray(raw[16:]), dtype=torch.uint8).clone()
    return data.view(-1, 1, 28, 28).float() / 255.0


def read_idx_labels(path: Path) -> torch.Tensor:
    """Read label tensor from an IDX gzip archive.

    Args:
        path: Path, compressed label archive.

    Returns:
        torch.Tensor, shape [N].
    """

    with gzip_open(path, "rb") as file_obj:
        raw = file_obj.read()
    return torch.frombuffer(bytearray(raw[8:]), dtype=torch.uint8).clone().long()


def load_datasets(validation_fraction: float, seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """Load FashionMNIST and create train/validation/test splits.

    Args:
        validation_fraction: float, validation split fraction.
        seed: int, split seed.

    Returns:
        tuple of train, validation, and test datasets.
    """

    data_dir = ensure_fashion_mnist_downloaded()
    train_images = read_idx_images(data_dir / "train-images-idx3-ubyte.gz")
    train_labels = read_idx_labels(data_dir / "train-labels-idx1-ubyte.gz")
    test_images = read_idx_images(data_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = read_idx_labels(data_dir / "t10k-labels-idx1-ubyte.gz")

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
    """Create train/validation/test dataloaders.

    Args:
        train_dataset: Dataset, training set.
        val_dataset: Dataset, validation set.
        test_dataset: Dataset, test set.
        batch_size: int, batch size.

    Returns:
        tuple of dataloaders.
    """

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Create one-hot encoded labels.

    Args:
        labels: torch.Tensor, shape [batch_size].
        num_classes: int, number of classes.

    Returns:
        torch.Tensor, shape [batch_size, num_classes].
    """

    return F.one_hot(labels, num_classes=num_classes).float()


def apply_mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Blend samples and labels using MixUp.

    Args:
        images: torch.Tensor, shape [batch_size, 1, 28, 28].
        labels: torch.Tensor, shape [batch_size].
        alpha: float, symmetric Beta distribution parameter.

    Returns:
        tuple of mixed images, mixed soft labels, lambda, and shuffled indices.
    """

    permutation = torch.randperm(images.size(0), device=images.device)
    lam = random.betavariate(alpha, alpha)
    labels_a = one_hot(labels, NUM_CLASSES)
    labels_b = one_hot(labels[permutation], NUM_CLASSES)
    mixed_images = lam * images + (1.0 - lam) * images[permutation]
    mixed_labels = lam * labels_a + (1.0 - lam) * labels_b
    return mixed_images, mixed_labels, lam, permutation


def clean_accuracy_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate top-1 accuracy on clean, unmixed labels.

    Args:
        model: nn.Module, trained model.
        loader: DataLoader, evaluation data.
        device: torch.device, target device.

    Returns:
        float, mean accuracy.
    """

    model.train(False)
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)
    return total_correct / total_examples


def smoothed_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    """Compute custom label-smoothed cross-entropy on soft targets.

    Args:
        logits: torch.Tensor, shape [batch_size, num_classes].
        targets: torch.Tensor, shape [batch_size, num_classes] or [batch_size].
        smoothing: float, smoothing coefficient.

    Returns:
        torch.Tensor scalar loss.
    """

    if targets.ndim == 1:
        targets = one_hot(targets, logits.size(1))
    smoothed_targets = (1.0 - smoothing) * targets + smoothing / logits.size(1)
    log_probabilities = torch.log_softmax(logits, dim=1)
    return -(smoothed_targets * log_probabilities).sum(dim=1).mean()


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy against hard labels.

    Args:
        logits: torch.Tensor, shape [batch_size, num_classes].
        labels: torch.Tensor, shape [batch_size].

    Returns:
        float, batch accuracy.
    """

    return float((logits.argmax(dim=1) == labels).float().mean().item())


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    smoothing: float,
    mixup_alpha: float,
    device: torch.device,
) -> Tuple[float, float]:
    """Train one epoch with MixUp and smoothed loss.

    Args:
        model: nn.Module, trainable model.
        loader: DataLoader, training batches.
        optimizer: Optimizer, parameter update rule.
        smoothing: float, label smoothing coefficient.
        mixup_alpha: float, Beta distribution parameter.
        device: torch.device, target device.

    Returns:
        tuple(float, float), mean loss and a proxy accuracy on original labels.
    """

    model.train(True)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        mixed_images, mixed_targets, _, _ = apply_mixup(images, labels, mixup_alpha)
        optimizer.zero_grad()
        logits = model(mixed_images)
        loss = smoothed_cross_entropy(logits, mixed_targets, smoothing)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    smoothing: float,
    device: torch.device,
    noise_std: float = 0.0,
) -> Tuple[float, float]:
    """Evaluate the model, optionally on a noisy input distribution.

    Args:
        model: nn.Module, trained model.
        loader: DataLoader, evaluation data.
        smoothing: float, label smoothing coefficient.
        device: torch.device, target device.
        noise_std: float, Gaussian noise standard deviation.

    Returns:
        tuple(float, float), mean loss and accuracy.
    """

    model.train(False)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if noise_std > 0.0:
                noise = torch.randn_like(images) * noise_std
                images = torch.clamp(images + noise, 0.0, 1.0)
            logits = model(images)
            loss = smoothed_cross_entropy(logits, labels, smoothing)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def save_model(
    model: nn.Module,
    config: Dict[str, float | int | str],
    best_epoch: int,
) -> None:
    """Save the final Task 2 checkpoint.

    Args:
        model: nn.Module, fitted model.
        config: dict, training configuration.
        best_epoch: int, best epoch selected by early stopping.
    """

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "best_epoch": best_epoch,
        },
        MODEL_PATH,
    )


def load_model(device: torch.device) -> Tuple[nn.Module, Dict[str, object]]:
    """Load the Task 2 checkpoint.

    Args:
        device: torch.device, target device.

    Returns:
        tuple(nn.Module, dict), model and stored metadata.
    """

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = build_model()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def save_history(history: History) -> None:
    """Write Task 2 history to JSON.

    Args:
        history: History, metrics container.
    """

    HISTORY_PATH.write_text(json.dumps(asdict(history), indent=2), encoding="utf-8")


def load_history() -> History:
    """Read Task 2 history from JSON."""

    return History(**json.loads(HISTORY_PATH.read_text(encoding="utf-8")))


def describe_soft_target(target: torch.Tensor) -> str:
    """Convert a soft label vector into a compact textual description.

    Args:
        target: torch.Tensor, shape [NUM_CLASSES].

    Returns:
        str, top-2 class mix summary.
    """

    values, indices = torch.topk(target, k=2)
    first = f"{CLASS_NAMES[int(indices[0])]} {float(values[0]):.2f}"
    second = f"{CLASS_NAMES[int(indices[1])]} {float(values[1]):.2f}"
    return f"{first} + {second}"
