"""
GenAI usage statement:
- Tool used: OpenAI Codex.
- Assistance received: scaffolding, implementation drafting, and code review.
- Human verification: the final code structure, hyperparameters, and task-specific logic
  were checked and adjusted to satisfy the coursework specification.
"""

from __future__ import annotations

import random
from typing import List

import torch
from PIL import Image, ImageDraw, ImageFont

from common import (
    CLASS_NAMES,
    ROBUSTNESS_PLOT_PATH,
    apply_mixup,
    ensure_directories,
    evaluate_epoch,
    load_datasets,
    load_history,
    load_model,
    make_loaders,
)


SEED = 13
DEVICE = torch.device("cpu")
BATCH_SIZE = 128
VALIDATION_FRACTION = 0.1
NOISE_STD = 0.25


def _montage_fonts() -> tuple:
    """Load fonts for the montage, falling back to the bitmap default.

    Returns:
        tuple of (title_font, label_font, sub_font).
    """

    try:
        return (
            ImageFont.load_default(size=22),
            ImageFont.load_default(size=14),
            ImageFont.load_default(size=12),
        )
    except TypeError:
        fb = ImageFont.load_default()
        return fb, fb, fb


def tensor_to_tile(image: torch.Tensor, tile_size: int = 168) -> Image.Image:
    """Convert one grayscale tensor image into a scaled RGB tile.

    Uses nearest-neighbour upscaling to preserve the pixel grid character of
    the FashionMNIST images.

    Args:
        image: torch.Tensor, shape [1, 28, 28].
        tile_size: int, output edge length in pixels.

    Returns:
        PIL.Image.Image, RGB tile of shape (tile_size, tile_size).
    """

    pixels = (image.squeeze(0).clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    base = Image.fromarray(pixels, mode="L").convert("RGB")
    return base.resize((tile_size, tile_size), Image.Resampling.NEAREST)


def save_mixup_montage(train_dataset) -> None:
    """Save a large, clearly labelled 4x4 montage of MixUp-blended images.

    Each cell shows one convex blend of two FashionMNIST training images.
    The label below each tile names the two constituent classes and their
    mixing weights (lambda from the Beta distribution draw).

    Args:
        train_dataset: Dataset, source split used for MixUp sampling.
    """

    rng = random.Random(SEED)
    indices = list(range(len(train_dataset)))
    rng.shuffle(indices)
    selected = indices[:16]

    images: List[torch.Tensor] = []
    labels: List[int] = []
    for index in selected:
        image, label = train_dataset[index]
        images.append(image)
        labels.append(label)

    batch_images = torch.stack(images, dim=0)
    batch_labels = torch.tensor(labels, dtype=torch.long)
    mixed_images, mixed_targets, lam, permutation = apply_mixup(
        batch_images,
        batch_labels,
        alpha=0.4,
    )

    COLS, ROWS = 4, 4
    TILE = 168          # px per image tile
    PAD = 12            # gap between tiles
    LABEL_H = 54        # height of text band below each tile
    BORDER = 2          # tile border thickness
    OUTER = 28          # canvas outer margin
    TITLE_H = 68        # top title area

    cell_w = TILE + 2 * BORDER
    cell_h = TILE + 2 * BORDER + LABEL_H

    canvas_w = OUTER * 2 + COLS * cell_w + (COLS - 1) * PAD
    canvas_h = OUTER + TITLE_H + ROWS * cell_h + (ROWS - 1) * PAD + OUTER

    BG = (248, 246, 240)
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    font_title, font_label, font_sub = _montage_fonts()

    # Title
    draw.text(
        (OUTER, OUTER),
        "MixUp Blending Montage — FashionMNIST Training Samples",
        fill=(20, 20, 20),
        font=font_title,
    )
    draw.text(
        (OUTER, OUTER + 30),
        "Each tile is a convex combination of two images: lambda*A + (1-lambda)*B",
        fill=(90, 90, 90),
        font=font_sub,
    )

    tile_origin_y = OUTER + TITLE_H

    for idx in range(COLS * ROWS):
        row = idx // COLS
        col = idx % COLS
        cell_x = OUTER + col * (cell_w + PAD)
        cell_y = tile_origin_y + row * (cell_h + PAD)

        # Border rectangle around image
        draw.rectangle(
            [(cell_x, cell_y), (cell_x + cell_w - 1, cell_y + TILE + 2 * BORDER - 1)],
            outline=(160, 155, 145),
            width=BORDER,
        )

        # Paste the image tile inside the border
        tile_img = tensor_to_tile(mixed_images[idx], TILE)
        canvas.paste(tile_img, (cell_x + BORDER, cell_y + BORDER))

        # Label band background
        label_y = cell_y + TILE + 2 * BORDER
        draw.rectangle(
            [(cell_x, label_y), (cell_x + cell_w - 1, label_y + LABEL_H - 1)],
            fill=(238, 234, 226),
            outline=(160, 155, 145),
            width=1,
        )

        # Split "ClassA w1 + ClassB w2" into two lines
        label_a = CLASS_NAMES[int(batch_labels[idx])]
        label_b = CLASS_NAMES[int(batch_labels[permutation[idx]])]
        weight_a = lam
        weight_b = 1.0 - lam
        if label_a == label_b:
            line1 = f"{label_a} 1.00"
            line2 = ""
        else:
            line1 = f"{label_a} {weight_a:.2f}"
            line2 = f"+ {label_b} {weight_b:.2f}"
        draw.text((cell_x + 5, label_y + 5), line1, fill=(25, 25, 25), font=font_label)
        draw.text((cell_x + 5, label_y + 24), line2, fill=(80, 75, 65), font=font_sub)

    # Index number badge on each tile (top-left corner)
    for idx in range(COLS * ROWS):
        row = idx // COLS
        col = idx % COLS
        cell_x = OUTER + col * (cell_w + PAD)
        cell_y = tile_origin_y + row * (cell_h + PAD)
        badge_x = cell_x + BORDER + 4
        badge_y = cell_y + BORDER + 4
        badge_r = [(badge_x, badge_y), (badge_x + 22, badge_y + 18)]
        draw.rectangle(badge_r, fill=(40, 40, 40))
        draw.text((badge_x + 3, badge_y + 1), str(idx + 1), fill=(255, 255, 255), font=font_sub)

    canvas.save(ROBUSTNESS_PLOT_PATH)


def report_text(
    clean_accuracy: float,
    noisy_accuracy: float,
    best_epoch: int,
    smoothing: float,
) -> str:
    """Create the required terminal report for Task 2.

    Args:
        clean_accuracy: float, clean test-set accuracy.
        noisy_accuracy: float, noisy test-set accuracy.
        best_epoch: int, selected epoch from early stopping.
        smoothing: float, label smoothing coefficient used during training.

    Returns:
        str, printable report.
    """

    return f"""
Task 2 technical justification

MixUp reduces memorization because it changes the learning problem from fitting isolated
training points to fitting linear relationships between points. Instead of repeatedly seeing
one image with one perfectly hard target, the model is shown convex combinations of two
images and must predict the matching convex combination of labels. That simple tensor-level
operation widens the support of the training distribution and removes the possibility of
treating each example as a disconnected lookup-table entry. If the network tries to memorize
one endpoint too aggressively, the loss on interpolated samples exposes that weakness because
the representation must remain meaningful along the path between classes. This encourages
smoother decision boundaries and discourages extremely sharp transitions that only work on the
original pixels. In practical terms, the model is rewarded for learning shared factors such as
shape, edges, and texture patterns instead of memorizing specific garments.

The training logs now report clean training accuracy computed on ordinary, unmixed training
examples after each epoch. That makes the train-validation comparison interpretable: both
numbers are measured on hard labels without MixUp applied at evaluation time, so a large gap
really does indicate overfitting rather than an artefact of the metric definition.

This is closely related to robustness. Interpolation-based training acts like a geometric prior:
nearby points in input space should produce nearby predictions in label space. That prior makes
the classifier less sensitive to small nuisance perturbations and class-specific artifacts. In
my run, the clean test accuracy was {clean_accuracy:.4f}, while the noisy-test accuracy with Gaussian
noise standard deviation {NOISE_STD:.2f} was {noisy_accuracy:.4f}. The drop is expected because the noisy
distribution is harder, but the model remains functional because MixUp has already trained it on
non-trivial mixtures rather than only pristine training examples. The saved montage (robustness_demo.png)
illustrates the MixUp mechanism: each of the 16 displayed samples is a convex blend of two
training images, with the label caption showing the two constituent classes and their mixing
weights. The network trained on such samples cannot rely on memorizing single pixels; it must
build representations that are meaningful along the interpolation path between classes.

Label smoothing complements MixUp by changing the optimization target. Standard cross-entropy
with one-hot labels pushes the logit of the target class upward until the predicted probability
approaches one, which can create very large margins and overconfident outputs. That behaviour
often leads to overshooting during optimization: updates continue to increase already-dominant
logits even when the sample is effectively solved. My custom loss replaces hard targets with
softened targets (epsilon={smoothing}), so the model is still encouraged to rank the correct class
highest, but it is not rewarded for collapsing the entire probability mass onto a single class.
This reduces logit extremes, moderates gradient magnitudes on easy samples, and makes the
optimization trajectory less brittle.

The combination is technically coherent. MixUp smooths the data manifold from the input side,
while label smoothing softens the supervision from the output side. Early stopping then selects
the checkpoint before validation performance begins to degrade; in this run the retained model
came from epoch {best_epoch}. Together these choices target memorization, overconfidence, and
overtraining with mutually reinforcing regularization signals rather than relying on any single
trick in isolation.
""".strip()


def main() -> None:
    """Load Task 2 artifacts, evaluate robustness, and save the montage."""

    ensure_directories()
    history = load_history()
    model, checkpoint = load_model(DEVICE)

    train_dataset, val_dataset, test_dataset = load_datasets(
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
    )
    _, _, test_loader = make_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=BATCH_SIZE,
    )

    config = checkpoint["config"]
    smoothing = float(config["smoothing"])

    clean_loss, clean_accuracy = evaluate_epoch(
        model=model,
        loader=test_loader,
        smoothing=smoothing,
        device=DEVICE,
        noise_std=0.0,
    )
    noisy_loss, noisy_accuracy = evaluate_epoch(
        model=model,
        loader=test_loader,
        smoothing=smoothing,
        device=DEVICE,
        noise_std=NOISE_STD,
    )

    save_mixup_montage(train_dataset)

    print(f"Saved montage to {ROBUSTNESS_PLOT_PATH}")
    print(f"Best epoch: {history.best_epoch}")
    print(f"Clean test  | loss={clean_loss:.4f}, accuracy={clean_accuracy:.4f}")
    print(f"Noisy test  | loss={noisy_loss:.4f}, accuracy={noisy_accuracy:.4f}\n")
    print(report_text(
        clean_accuracy=clean_accuracy,
        noisy_accuracy=noisy_accuracy,
        best_epoch=history.best_epoch,
        smoothing=smoothing,
    ))


if __name__ == "__main__":
    main()
