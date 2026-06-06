"""Generate training loss figures from W&B-exported CSV logs.

Form1 plots use prompted ScanNet training (``feature_head_sam_prompted_scannet``).
W&B CSV columns ``loss:rgb_mse`` / ``loss:rgb_lpips`` hold **already-weighted**
``loss/mse`` and ``loss/lpips`` scalars from ``ModelWrapper.training_step``.
Coefficients come from ``config/loss/mse.yaml`` and ``config/loss/lpips.yaml``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

STEP_COLUMN = "trainer/global_step"
LINE_ALPHA = 0.7
TITLE_FONTSIZE = plt.rcParams["axes.titlesize"]
TICK_FONTSIZE = plt.rcParams["xtick.labelsize"]

# Weight applied inside LossMse / LossLpips before logging (see config/loss/*.yaml).
RGB_MSE_COEF = 1.0
RGB_LPIPS_COEF = 0.05
RGB_LOSS_TITLE = f"RGB Loss ({RGB_MSE_COEF}×MSE + {RGB_LPIPS_COEF}×LPIPS)"


def _load_loss_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    value_columns = [
        col
        for col in df.columns
        if col != STEP_COLUMN
        and not col.endswith("__MIN")
        and not col.endswith("__MAX")
        and " - _step" not in col
    ]
    if not value_columns:
        raise ValueError(f"No loss value column found in {path}")
    return df[[STEP_COLUMN, value_columns[0]]].rename(
        columns={value_columns[0]: "value"}
    )


def _plot_loss(
    ax: plt.Axes,
    steps: pd.Series,
    values: pd.Series,
    *,
    title: str,
    color: str,
) -> None:
    ax.plot(steps, values, linewidth=1.5, color=color, alpha=LINE_ALPHA)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)


def plot_loss_figure(data_dir: Path, output_path: Path) -> None:
    total = _load_loss_csv(data_dir / "loss:total.csv")
    rgb_lpips = _load_loss_csv(data_dir / "loss:rgb_lpips.csv")
    rgb_mse = _load_loss_csv(data_dir / "loss:rgb_mse.csv")
    feature = _load_loss_csv(data_dir / "loss:feature.csv")
    seg = _load_loss_csv(data_dir / "loss:seg.csv")

    rgb = rgb_lpips.merge(rgb_mse, on=STEP_COLUMN, suffixes=("_lpips", "_mse"))
    rgb["value"] = rgb["value_lpips"] + rgb["value_mse"]

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.35, wspace=0.25)

    ax_total = fig.add_subplot(gs[0, :])
    ax_rgb = fig.add_subplot(gs[1, 0])
    ax_feature = fig.add_subplot(gs[1, 1], sharex=ax_rgb)
    ax_seg = fig.add_subplot(gs[1, 2], sharex=ax_rgb)

    _plot_loss(ax_total, total[STEP_COLUMN], total["value"], title="Total Loss", color="brown")
    _plot_loss(ax_rgb, rgb[STEP_COLUMN], rgb["value"], title=RGB_LOSS_TITLE, color="red")
    _plot_loss(
        ax_feature, feature[STEP_COLUMN], feature["value"], title="Feature Loss", color="green"
    )
    _plot_loss(
        ax_seg, seg[STEP_COLUMN], seg["value"], title="Segmentation Loss", color="blue"
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_form1_figures(data_dir: Path | None = None) -> Path:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "c3gsam_results" / "form1"

    data_dir = data_dir.resolve()
    output_path = data_dir / "loss.png"
    plot_loss_figure(data_dir, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate form1 training loss figures.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing loss:*.csv files (default: c3gsam_results/form1).",
    )
    args = parser.parse_args()

    output_path = generate_form1_figures(args.data_dir)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
