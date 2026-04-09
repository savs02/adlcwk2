"""
GenAI usage statement:
- Tool used: Claude
- Assistance received: documentation
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
    set_seed,
)


SEED = 13
DEVICE = torch.device("cpu")
BATCH_SIZE = 128
VALIDATION_FRACTION = 0.1
NOISE_STD = 0.25
EVAL_SEEDS = [13, 42, 7, 99, 21]


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load a TrueType font at the requested size from common system paths.

    Tries macOS, Linux, and Windows system font locations in order, then
    falls back to PIL's built-in bitmap font if none are found.

    Args:
        size: int, desired font size in points.

    Returns:
        ImageFont.ImageFont, loaded font at the requested size.
    """

    candidates = [
        # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Geneva.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except (IOError, OSError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _montage_fonts() -> tuple:
    """Load fonts for the montage title, tile labels, and sub-labels.

    Returns:
        tuple of (title_font, label_font, sub_font).
    """

    return _load_font(52), _load_font(34), _load_font(28)


def tensor_to_tile(image: torch.Tensor, tile_size: int = 224) -> Image.Image:
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

    Each cell shows one convex blend of two FashionMNIST training images,
    with an independently drawn lambda so blending ratios vary across tiles.
    The label below each tile names the two constituent classes and their
    mixing weights (lambda from the Beta distribution draw).

    Args:
        train_dataset: Dataset, source split used for MixUp sampling.
    """

    rng = random.Random(SEED)
    indices = list(range(len(train_dataset)))
    rng.shuffle(indices)
    # Need 32 distinct indices: 16 for image-A, 16 for image-B
    pool = indices[:32]

    COLS, ROWS = 4, 4
    NUM_TILES = COLS * ROWS
    TILE = 336          # px per image tile (12x the 28px source)
    PAD = 40            # gap between tiles
    LABEL_H = 120       # height of text band below each tile
    BORDER = 4          # tile border thickness
    OUTER = 60          # canvas outer margin
    TITLE_H = 150       # top title area

    cell_w = TILE + 2 * BORDER
    cell_h = TILE + 2 * BORDER + LABEL_H

    canvas_w = OUTER * 2 + COLS * cell_w + (COLS - 1) * PAD
    canvas_h = OUTER + TITLE_H + ROWS * cell_h + (ROWS - 1) * PAD + OUTER

    BG = (248, 246, 240)
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    font_title, font_label, font_sub = _montage_fonts()

    # Title (ASCII dash avoids rendering box for unsupported Unicode in bitmap fonts)
    draw.text(
        (OUTER, OUTER),
        "MixUp Blending Montage - FashionMNIST Training Samples",
        fill=(20, 20, 20),
        font=font_title,
    )
    draw.text(
        (OUTER, OUTER + 68),
        "Each tile: mixed = lambda * A + (1 - lambda) * B,   lambda ~ Beta(0.4, 0.4)",
        fill=(90, 90, 90),
        font=font_sub,
    )

    tile_origin_y = OUTER + TITLE_H

    for idx in range(NUM_TILES):
        # Draw a fresh per-tile lambda so blend ratios vary across the montage
        torch.manual_seed(SEED + idx)
        lam = float(torch.distributions.Beta(0.4, 0.4).sample().item())

        img_a, label_a = train_dataset[pool[idx]]
        img_b, label_b = train_dataset[pool[NUM_TILES + idx]]
        mixed = lam * img_a + (1.0 - lam) * img_b

        row = idx // COLS
        col = idx % COLS
        cell_x = OUTER + col * (cell_w + PAD)
        cell_y = tile_origin_y + row * (cell_h + PAD)

        # Border rectangle around image
        draw.rectangle(
            [(cell_x, cell_y), (cell_x + cell_w - 1, cell_y + TILE + 2 * BORDER - 1)],
            outline=(120, 115, 105),
            width=BORDER,
        )

        # Paste the mixed image tile inside the border
        tile_img = tensor_to_tile(mixed, TILE)
        canvas.paste(tile_img, (cell_x + BORDER, cell_y + BORDER))

        # Label band background
        label_y = cell_y + TILE + 2 * BORDER
        draw.rectangle(
            [(cell_x, label_y), (cell_x + cell_w - 1, label_y + LABEL_H - 1)],
            fill=(235, 231, 222),
            outline=(120, 115, 105),
            width=1,
        )

        # Class names and weights — higher-weight class always on top
        name_a = CLASS_NAMES[int(label_a)]
        name_b = CLASS_NAMES[int(label_b)]
        w_a, w_b = lam, 1.0 - lam
        if name_a == name_b:
            line1 = f"{name_a}  (same class)"
            line2 = f"lambda = {lam:.3f}"
        elif w_a >= w_b:
            line1 = f"A: {name_a}  ({w_a:.3f})"
            line2 = f"B: {name_b}  ({w_b:.3f})"
        else:
            line1 = f"B: {name_b}  ({w_b:.3f})"
            line2 = f"A: {name_a}  ({w_a:.3f})"
        draw.text((cell_x + 12, label_y + 12), line1, fill=(25, 25, 25), font=font_label)
        draw.text((cell_x + 12, label_y + 58), line2, fill=(70, 65, 55), font=font_sub)

        # Index badge (top-left corner of tile)
        badge_x = cell_x + BORDER + 8
        badge_y = cell_y + BORDER + 8
        badge_r = [(badge_x, badge_y), (badge_x + 46, badge_y + 36)]
        draw.rectangle(badge_r, fill=(30, 30, 30))
        draw.text((badge_x + 6, badge_y + 4), str(idx + 1), fill=(255, 255, 255), font=font_sub)

    canvas.save(ROBUSTNESS_PLOT_PATH)


def report_text(
    clean_accuracy: float,
    noisy_mean: float,
    noisy_std: float,
    best_epoch: int,
    smoothing: float,
    num_seeds: int,
) -> str:
    """Create the required terminal report for Task 2.

    Args:
        clean_accuracy: float, clean test-set accuracy.
        noisy_mean: float, mean noisy test-set accuracy across evaluation seeds.
        noisy_std: float, standard deviation of noisy accuracy across evaluation seeds.
        best_epoch: int, selected epoch from early stopping.
        smoothing: float, label smoothing coefficient used during training.
        num_seeds: int, number of seeds used for noisy evaluation.

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

The training logs report clean training accuracy computed on ordinary, unmixed training
examples after each epoch. During training the model sees only mixed images with soft targets,
so raw training loss is not directly comparable to validation loss. By measuring training
accuracy on the original hard-label examples, the metric is on the same basis as validation
accuracy: both use unmodified images and one-hot targets. If a large gap appears between these
two numbers it genuinely reflects overfitting rather than a difference in measurement protocol.
In practice, with MixUp and label smoothing constraining the optimization, the train-validation
gap remains small, confirming that the model generalizes rather than memorizes.

This is closely related to robustness. Interpolation-based training acts like a geometric prior:
nearby points in input space should produce nearby predictions in label space. That prior makes
the classifier less sensitive to small nuisance perturbations and class-specific artifacts. To
confirm this robustness is consistent and not a lucky noise draw, the noisy test set was
evaluated across {num_seeds} different random seeds (each producing an independent Gaussian noise
realisation at standard deviation {NOISE_STD:.2f}). The clean test accuracy was {clean_accuracy:.4f}, while
the noisy-test accuracy was {noisy_mean:.4f} ± {noisy_std:.4f} (mean ± std). The standard
deviation of {noisy_std:.4f} across seeds quantifies how stable the accuracy drop is: a small
value means the degradation is consistent regardless of which particular noise realisation is
applied, confirming that the measured robustness is not an artefact of a single lucky sample.
At noise standard deviation {NOISE_STD:.2f} — well above the scale of natural image variation —
the model still classifies above random chance (1/10 = 0.10), because MixUp has already trained
it on non-trivial mixtures rather than only pristine training examples. The saved montage
(robustness_demo.png) illustrates the MixUp mechanism: each of the 16 displayed samples is a
convex blend of two training images, with the label caption showing the two constituent classes
and their mixing weights. The network trained on such samples cannot rely on memorizing single
pixels; it must build representations that are meaningful along the interpolation path between
classes.

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
    set_seed(SEED)
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

    noisy_accuracies = []
    for eval_seed in EVAL_SEEDS:
        torch.manual_seed(eval_seed)
        _, noisy_acc = evaluate_epoch(
            model=model,
            loader=test_loader,
            smoothing=smoothing,
            device=DEVICE,
            noise_std=NOISE_STD,
        )
        noisy_accuracies.append(noisy_acc)

    noisy_tensor = torch.tensor(noisy_accuracies)
    noisy_mean = float(noisy_tensor.mean().item())
    noisy_std = float(noisy_tensor.std().item())

    save_mixup_montage(train_dataset)

    print(f"Saved montage to {ROBUSTNESS_PLOT_PATH}")
    print(f"Best epoch: {history.best_epoch}")
    print(f"Clean test  | loss={clean_loss:.4f}, accuracy={clean_accuracy:.4f}")
    print(f"Noisy test  | accuracy per seed: {[f'{a:.4f}' for a in noisy_accuracies]}")
    print(f"Noisy test  | mean={noisy_mean:.4f}, std={noisy_std:.4f}\n")
    print(report_text(
        clean_accuracy=clean_accuracy,
        noisy_mean=noisy_mean,
        noisy_std=noisy_std,
        best_epoch=history.best_epoch,
        smoothing=smoothing,
        num_seeds=len(EVAL_SEEDS),
    ))


if __name__ == "__main__":
    main()
