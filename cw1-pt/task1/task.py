"""
GenAI usage statement:
- Tool used: Claude
- Assistance received: documentation
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from common import (
    BASELINE_MODEL_PATH,
    PLOT_PATH,
    REGULARIZED_MODEL_PATH,
    ensure_directories,
    evaluate_model,
    load_datasets,
    load_history,
    load_model,
    make_loaders,
)


SEED = 7
BATCH_SIZE = 128
VALIDATION_FRACTION = 0.1
DEVICE = torch.device("cpu")


def _load_fonts() -> Tuple[object, object, object, object]:
    """Load PIL fonts at several sizes, falling back to the bitmap default.

    Returns:
        tuple of (title_font, axis_font, tick_font, legend_font).
    """

    sizes = (26, 20, 15, 17)
    try:
        return tuple(ImageFont.load_default(size=s) for s in sizes)  # type: ignore[return-value]
    except TypeError:
        fallback = ImageFont.load_default()
        return fallback, fallback, fallback, fallback


def _dashed_segment(
    draw: ImageDraw.ImageDraw,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    fill: Tuple[int, int, int],
    width: int,
    dash: int = 16,
    gap: int = 9,
) -> None:
    """Draw a single dashed line segment between two points.

    Args:
        draw: ImageDraw.ImageDraw, drawing context.
        x0: float, start x.
        y0: float, start y.
        x1: float, end x.
        y1: float, end y.
        fill: tuple[int, int, int], RGB colour.
        width: int, stroke width in pixels.
        dash: int, length of each drawn dash in pixels.
        gap: int, length of each gap in pixels.
    """

    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1:
        return
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    pos, pen_down = 0.0, True
    while pos < length:
        seg_len = dash if pen_down else gap
        nxt = min(pos + seg_len, length)
        if pen_down:
            draw.line(
                [(x0 + dx * pos, y0 + dy * pos), (x0 + dx * nxt, y0 + dy * nxt)],
                fill=fill,
                width=width,
            )
        pos = nxt
        pen_down = not pen_down


def _draw_series(
    draw: ImageDraw.ImageDraw,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    line_width: int,
    dot_radius: int,
    dashed: bool,
    bg: Tuple[int, int, int],
) -> None:
    """Draw one data series as a polyline with per-point markers.

    Args:
        draw: ImageDraw.ImageDraw, drawing context.
        points: list[tuple[int, int]], pixel coordinates for each epoch.
        color: tuple[int, int, int], series RGB colour.
        line_width: int, polyline stroke width.
        dot_radius: int, marker radius in pixels.
        dashed: bool, if True the line is dashed and markers are open circles.
        bg: tuple[int, int, int], canvas background colour for open markers.
    """

    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if dashed:
            _dashed_segment(draw, x0, y0, x1, y1, color, line_width)
        else:
            draw.line([(x0, y0), (x1, y1)], fill=color, width=line_width)

    r = dot_radius
    for px, py in points:
        bbox = [(px - r, py - r), (px + r, py + r)]
        if dashed:
            draw.ellipse(bbox, outline=color, fill=bg, width=2)
        else:
            draw.ellipse(bbox, fill=color)


def draw_plot(histories: Dict[str, object], output_path: str) -> None:
    """Create the required training-versus-validation accuracy PNG.

    The plot is sized for easy reading: zoomed Y-axis, per-point markers,
    solid lines for training curves, dashed lines for validation curves,
    and a labelled legend box.

    Args:
        histories: dict[str, History], training metrics keyed by model name.
        output_path: str, destination PNG path.
    """

    # canvas geometry 
    W, H = 1800, 1050
    L_MAR, R_MAR, T_MAR, B_MAR = 125, 330, 100, 95
    plot_w = W - L_MAR - R_MAR
    plot_h = H - T_MAR - B_MAR
    BG = (252, 250, 246)

    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    font_title, font_axis, font_tick, font_legend = _load_fonts()

    # y-axis range: zoom to the actual data 
    all_vals: List[float] = (
        histories["baseline"].train_accuracy
        + histories["baseline"].val_accuracy
        + histories["regularized"].train_accuracy
        + histories["regularized"].val_accuracy
    )
    raw_lo = min(all_vals)
    raw_hi = max(all_vals)
    y_lo = max(0.0, round(raw_lo / 0.05) * 0.05 - 0.05)
    y_hi = min(1.0, round(raw_hi / 0.05) * 0.05 + 0.05)
    y_range = y_hi - y_lo

    def to_px(epoch_idx: int, value: float, n_epochs: int) -> Tuple[int, int]:
        """Convert (epoch_index, accuracy_value) to canvas pixel coordinates."""
        x = L_MAR + int(epoch_idx * plot_w / max(n_epochs - 1, 1))
        y = T_MAR + int((1.0 - (value - y_lo) / y_range) * plot_h)
        return x, y

    # y grid lines and tick labels 
    tick = y_lo
    while tick <= y_hi + 1e-9:
        py = T_MAR + int((1.0 - (tick - y_lo) / y_range) * plot_h)
        is_major = round(tick * 100) % 10 == 0       # every 0.10
        grid_col = (195, 195, 195) if is_major else (220, 220, 220)
        grid_w = 2 if is_major else 1
        draw.line([(L_MAR, py), (L_MAR + plot_w, py)], fill=grid_col, width=grid_w)
        draw.text((L_MAR - 68, py - 10), f"{tick:.2f}", fill=(55, 55, 55), font=font_tick)
        tick = round(tick + 0.05, 10)

    # x grid lines and epoch labels
    epoch_count = len(histories["baseline"].train_accuracy)
    for ei in range(epoch_count):
        px = L_MAR + int(ei * plot_w / max(epoch_count - 1, 1))
        draw.line([(px, T_MAR), (px, T_MAR + plot_h)], fill=(228, 228, 228), width=1)
        label = str(ei + 1)
        draw.text((px - 6, T_MAR + plot_h + 14), label, fill=(55, 55, 55), font=font_tick)

    # plot border 
    draw.rectangle(
        [(L_MAR, T_MAR), (L_MAR + plot_w, T_MAR + plot_h)],
        outline=(70, 70, 70),
        width=2,
    )

    # data series 
    LINE_W = 5
    DOT_R = 8
    series = [
        ("Baseline train",     histories["baseline"].train_accuracy,     (210, 72, 66),  False),
        ("Baseline val",       histories["baseline"].val_accuracy,        (210, 72, 66),  True),
        ("Regularized train",  histories["regularized"].train_accuracy,   (55, 118, 200), False),
        ("Regularized val",    histories["regularized"].val_accuracy,     (55, 118, 200), True),
    ]
    for _name, values, color, dashed in series:
        n = len(values)
        pts = [to_px(i, v, n) for i, v in enumerate(values)]
        _draw_series(draw, pts, color, LINE_W, DOT_R, dashed, BG)

    # legend box 
    LX = L_MAR + plot_w + 22
    LY = T_MAR + 18
    LW, LH = 295, 195
    draw.rectangle([(LX, LY), (LX + LW, LY + LH)], fill=(244, 241, 236), outline=(110, 110, 110), width=1)
    legend_entries = [
        ("Baseline train",    (210, 72, 66),  False),
        ("Baseline val",      (210, 72, 66),  True),
        ("Regularized train", (55, 118, 200), False),
        ("Regularized val",   (55, 118, 200), True),
    ]
    for idx, (lbl, col, dashed) in enumerate(legend_entries):
        ey = LY + 22 + idx * 40
        ex = LX + 16
        swatch_y = ey + 9
        if dashed:
            _dashed_segment(draw, ex, swatch_y, ex + 38, swatch_y, col, 3)
            draw.ellipse([(ex + 15, swatch_y - 5), (ex + 25, swatch_y + 5)],
                         outline=col, fill=(244, 241, 236), width=2)
        else:
            draw.line([(ex, swatch_y), (ex + 38, swatch_y)], fill=col, width=3)
            draw.ellipse([(ex + 15, swatch_y - 5), (ex + 25, swatch_y + 5)], fill=col)
        draw.text((ex + 50, ey), lbl, fill=(25, 25, 25), font=font_legend)

    # axis labels 
    draw.text((L_MAR + plot_w // 2 - 30, H - 58), "Epoch", fill=(30, 30, 30), font=font_axis)

    acc_img = Image.new("RGB", (200, 28), BG)
    acc_draw = ImageDraw.Draw(acc_img)
    acc_draw.text((0, 4), "Accuracy", fill=(30, 30, 30), font=font_axis)
    rotated = acc_img.rotate(90, expand=True)
    canvas.paste(rotated, (8, T_MAR + plot_h // 2 - 100))

    # title and subtitle 
    draw.text(
        (L_MAR, 22),
        "Generalization Gap: Training vs Validation Accuracy (FashionMNIST)",
        fill=(20, 20, 20),
        font=font_title,
    )
    draw.text(
        (L_MAR, 60),
        "Solid line + filled marker = training    Dashed line + open marker = validation",
        fill=(100, 100, 100),
        font=font_tick,
    )

    canvas.save(output_path)


def technical_analysis(
    baseline_train_acc: float,
    baseline_val_acc: float,
    regularized_train_acc: float,
    regularized_val_acc: float,
    baseline_test_acc: float,
    regularized_test_acc: float,
    reg_best_val_acc: float,
    reg_best_epoch: int,
) -> str:
    """Return the coursework analysis text.

    Args:
        baseline_train_acc: float, final baseline training accuracy.
        baseline_val_acc: float, final baseline validation accuracy.
        regularized_train_acc: float, final regularized training accuracy.
        regularized_val_acc: float, final regularized validation accuracy.
        baseline_test_acc: float, baseline test accuracy.
        regularized_test_acc: float, regularized test accuracy.
        reg_best_val_acc: float, best validation accuracy seen during regularized training.
        reg_best_epoch: int, epoch at which regularized model achieved best validation accuracy.

    Returns:
        str, printable technical discussion.
    """

    baseline_gap = baseline_train_acc - baseline_val_acc
    regularized_gap = regularized_train_acc - regularized_val_acc
    gap_pct = (baseline_gap - regularized_gap) / baseline_gap * 100

    return f"""
Task 1 technical analysis

Both models use the same six-layer architecture (hidden widths 1024, 512, 512, 256, 128, 64
with ReLU activations), so the only experimental differences are the regularization applied to
the second model and the learning rate, which I had to lower from 0.08 to 0.05 for the
regularized model because dropout destabilized training at the higher rate.

The baseline overfits clearly. Training accuracy climbs across all 25 epochs, reaching
{baseline_train_acc:.4f}. Validation accuracy keeps up through the first 8 epochs but from
epoch 9 onward training accuracy is consistently above validation and the gap grows steadily
to {baseline_gap:.4f} by epoch 25. Looking at the plot, the validation curve flattens and
oscillates around 0.88-0.89 while training continues upward, and that persistent divergence is
the generalization gap. Test accuracy of {baseline_test_acc:.4f} lines up with the validation
result, which rules out the gap being an artifact of how the validation split landed.

The regularized model adds L2 weight decay (lambda=1e-4) and dropout at rate 0.15. I
originally tried dropout 0.25 and validation accuracy dropped well below the baseline, which
put the model in the high-bias region rather than the intended lower-variance position. At
0.15 the training accuracy ends at {regularized_train_acc:.4f}, lower than the baseline's
{baseline_train_acc:.4f} as expected since the constraints stop it fitting training noise as
freely. The train-val gap narrows to {regularized_gap:.4f}, about {gap_pct:.0f}% tighter than
the baseline's {baseline_gap:.4f}. What I did not expect is the regularized model's final
validation accuracy ({regularized_val_acc:.4f}) being slightly below the baseline's
({baseline_val_acc:.4f}). Its best validation result came at epoch {reg_best_epoch}
({reg_best_val_acc:.4f}), where the train-val gap was only 0.0064, but it drifted back by
epoch 25. The baseline happened to reach its peak validation accuracy right at the final epoch.
The gap reduction is real and the variance is clearly lower, but it did not translate into a
clean accuracy win here. I think 25 epochs is not enough for the regularized model to fully
converge at the lower learning rate, and with more epochs I would expect the gap to close.

On the optimizer, I tried Adam first and switched to SGD with momentum 0.9. Adam normalizes
each parameter's gradient by a running estimate of its variance, which stabilizes training but
removes most of the stochasticity from the update signal. That stochasticity in SGD mini-batch
estimates is itself a form of implicit regularization, meaning gradient noise is proportional to
learning rate divided by batch size, and it biases the optimizer away from sharp narrow minima
(where small weight perturbations sharply increase the loss) toward flatter regions that
generalize better. With LR=0.08 and batch size 128, the baseline's gradient noise scale is
roughly 6e-4 per update step and the regularized model at LR=0.05 sits at about 4e-4. The
explicit dropout and weight decay dominate the regularized model's behavior, but SGD's
gradient noise is present in both and is part of why neither model collapses into a sharp
minimum despite having enough capacity to memorize the training set.
""".strip()


def main() -> None:
    """Load Task 1 artifacts, generate the plot, and print analysis."""

    ensure_directories()
    histories = load_history()
    draw_plot(histories, str(PLOT_PATH))

    train_dataset, val_dataset, test_dataset = load_datasets(
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
    )
    _, val_loader, test_loader = make_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=BATCH_SIZE,
    )
    criterion = nn.CrossEntropyLoss()

    baseline_model, _ = load_model(BASELINE_MODEL_PATH, DEVICE)
    regularized_model, _ = load_model(REGULARIZED_MODEL_PATH, DEVICE)

    _, baseline_val_acc = evaluate_model(baseline_model, val_loader, criterion, DEVICE)
    _, regularized_val_acc = evaluate_model(regularized_model, val_loader, criterion, DEVICE)
    _, baseline_test_acc = evaluate_model(baseline_model, test_loader, criterion, DEVICE)
    _, regularized_test_acc = evaluate_model(regularized_model, test_loader, criterion, DEVICE)

    baseline_train_acc = histories["baseline"].train_accuracy[-1]
    regularized_train_acc = histories["regularized"].train_accuracy[-1]

    reg_val_list = histories["regularized"].val_accuracy
    reg_best_val_acc = max(reg_val_list)
    reg_best_epoch = reg_val_list.index(reg_best_val_acc) + 1

    print(f"Saved plot to {PLOT_PATH}")
    print(f"Training accuracy   | baseline={baseline_train_acc:.4f}, regularized={regularized_train_acc:.4f}")
    print(f"Validation accuracy | baseline={baseline_val_acc:.4f}, regularized={regularized_val_acc:.4f}")
    print(f"Test accuracy       | baseline={baseline_test_acc:.4f}, regularized={regularized_test_acc:.4f}\n")
    print(technical_analysis(
        baseline_train_acc=baseline_train_acc,
        baseline_val_acc=baseline_val_acc,
        regularized_train_acc=regularized_train_acc,
        regularized_val_acc=regularized_val_acc,
        baseline_test_acc=baseline_test_acc,
        regularized_test_acc=regularized_test_acc,
        reg_best_val_acc=reg_best_val_acc,
        reg_best_epoch=reg_best_epoch,
    ))


if __name__ == "__main__":
    main()
