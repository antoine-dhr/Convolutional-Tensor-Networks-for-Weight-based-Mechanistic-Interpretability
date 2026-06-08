import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from converter.test_convert import (
    cat_ch,
    ones_ch,
    scalar_ch,
    contract_model_stepwise,
)
from utils.RMSNorm2d import RMSNorm2d
from utils.ResidualBlock import ResidualBlock
from utils.model import BaseCNN


# Configuration for the experiment
H, W          = 8, 8
IN_CHANNELS   = 3
NUM_CLASSES   = 10
NUM_TRIALS    = 10

BLOCKS_PER_STAGE_VALUES = [1, 2, 3, 4, 5, 6, 7]
NUM_STAGES_VALUES       = [1, 2, 3, 4, 5, 6, 7]
REPEAT_FACTOR_VALUES    = [1, 2, 3]


def build_cfg(num_stages, blocks_per_stage, repeat_factor):
    class Cfg:
        pass
    cfg = Cfg()
    cfg.num_stages           = num_stages
    cfg.blocks_per_stage     = blocks_per_stage
    cfg.upscale_factor       = 2
    cfg.channels_factor      = 2
    cfg.start_output_channels = IN_CHANNELS
    cfg.kernel_size          = 3
    cfg.repeat_factor        = repeat_factor
    cfg.rms_norm_mode        = "global"
    return cfg


def randomize_gammas(model):
    for layer in model.layers:
        if isinstance(layer, ResidualBlock):
            for sub in layer.layers:
                if isinstance(sub, RMSNorm2d):
                    sub.gamma = nn.Parameter(torch.rand_like(sub.gamma.data))


def trial_mean_abs_error(num_stages, blocks_per_stage, repeat_factor):
    cfg   = build_cfg(num_stages, blocks_per_stage, repeat_factor)
    model = BaseCNN(cfg, in_channels=IN_CHANNELS,
                    input_size=H, num_classes=NUM_CLASSES)
    model.eval()
    randomize_gammas(model)

    data = torch.rand(IN_CHANNELS, H, W)
    inp  = cat_ch(ones_ch(H, W), scalar_ch(1.0, H, W), data, data)

    tn_out = contract_model_stepwise(model, inp)
    with torch.no_grad():
        expected = model(data.unsqueeze(0)).squeeze(0)

    output = tn_out[1:] / tn_out[0]
    return (output - expected).abs().mean().item()


def run_experiment_for_repeat(repeat_factor):
    rows = len(NUM_STAGES_VALUES)
    cols = len(BLOCKS_PER_STAGE_VALUES)
    grid = np.zeros((rows, cols))

    for i, num_stages in enumerate(NUM_STAGES_VALUES):
        for j, blocks_per_stage in enumerate(BLOCKS_PER_STAGE_VALUES):
            errors = [
                trial_mean_abs_error(num_stages, blocks_per_stage, repeat_factor)
                for _ in range(NUM_TRIALS)
            ]
            grid[i, j] = float(np.mean(errors))
            print(
                f"repeat={repeat_factor}  stages={num_stages}  "
                f"blocks={blocks_per_stage}  MAE={grid[i, j]:.3e}",
                flush=True,   # important: keeps SLURM logs live
            )
    return grid


# Plotting code based on https://seaborn.pydata.org/generated/seaborn.heatmap.html
def _scientific_notation(val):
    if not np.isfinite(val) or val == 0:
        return "0"
    mantissa, exponent = f"{val:.2e}".split("e")
    exponent = int(exponent)
    return rf"${mantissa} \times 10^{{{exponent}}}$"


def compute_global_range(grids):
    stacked = np.stack(grids)
    vmin = float(np.nanmin(stacked))
    vmax = float(np.nanmax(stacked))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1e-12
    return vmin, vmax


def plot_heatmap(grid, repeat_factor, out_path, vmin=None, vmax=None):
    sns.set_theme(context="paper", style="white", font_scale=1.1)

    annot = np.vectorize(_scientific_notation)(grid)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        grid,
        annot=annot,
        fmt="",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={
            "label": "Mean absolute error",
            "format": FuncFormatter(lambda x, _: _scientific_notation(x)),
        },
        linewidths=0.5,
        linecolor="white",
        square=True,
        xticklabels=BLOCKS_PER_STAGE_VALUES,
        yticklabels=NUM_STAGES_VALUES,
        annot_kws={"fontsize": 6},
        ax=ax,
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Blocks per stage")
    ax.set_ylabel("Number of stages")
    ax.text(
        0.5, 1.08, "Conversion accuracy",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=12, fontweight="bold",
    )
    ax.text(
        0.5, 1.02, f"(Repeat factor = {repeat_factor})",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}", flush=True)


def plot_heatmaps_panel(grids, repeat_factors, out_path, vmin, vmax):
    sns.set_theme(context="paper", style="white", font_scale=1.0)

    num_plots = len(grids)
    single_w = 4.8
    single_h = 4.6
    if num_plots == 3:
        fig = plt.figure(figsize=(2 * single_w + 2.4, 2 * single_h + 1.2))
        gs = fig.add_gridspec(
            2, 4,
            width_ratios=[1, 1, 1, 1],
            height_ratios=[1, 1],
            wspace=0.3,
            hspace=0.25,
        )
        axes = [
            fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[1, 1:3]),
        ]
    else:
        fig, axes = plt.subplots(1, num_plots, figsize=(single_w * num_plots, single_h))
        if num_plots == 1:
            axes = [axes]

    for ax, grid, repeat_factor in zip(axes, grids, repeat_factors):
        annot = np.vectorize(_scientific_notation)(grid)
        sns.heatmap(
            grid,
            annot=annot,
            fmt="",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            square=True,
            xticklabels=BLOCKS_PER_STAGE_VALUES,
            yticklabels=NUM_STAGES_VALUES,
            annot_kws={"fontsize": 5},
            ax=ax,
        )
        ax.invert_yaxis()
        ax.set_xlabel("Blocks per stage")
        ax.set_ylabel("Number of stages")
        ax.set_title(f"Repeat factor = {repeat_factor}", fontsize=10)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.2, 0.02, 0.6])
    norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        cax=cbar_ax,
        format=FuncFormatter(lambda x, _: _scientific_notation(x)),
    )
    cbar.set_label("Mean absolute error")
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}", flush=True)


def main():
    out_dir = Path(__file__).resolve().parent / "scalability_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    grids = []
    for repeat_factor in REPEAT_FACTOR_VALUES:
        print(f"\n=== Repeat factor {repeat_factor} ===", flush=True)
        grid = run_experiment_for_repeat(repeat_factor)
        grids.append(grid)
        np.save(out_dir / f"grid_repeat_factor_{repeat_factor}_hydra.npy", grid)

    vmin, vmax = compute_global_range(grids)

    for repeat_factor, grid in zip(REPEAT_FACTOR_VALUES, grids):
        out_path = out_dir / f"heatmap_repeat_factor_{repeat_factor}_hydra.png"
        plot_heatmap(grid, repeat_factor, out_path, vmin=vmin, vmax=vmax)

    panel_path = out_dir / "heatmap_repeat_factors_panel_hydra.png"
    plot_heatmaps_panel(grids, REPEAT_FACTOR_VALUES, panel_path, vmin, vmax)


if __name__ == "__main__":
    main()
