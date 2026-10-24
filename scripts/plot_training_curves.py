"""Per-epoch curves of runs trained with early stopping (--val_ratio): validation loss on held-out
normal train data and test AUROC, one line per run, one row per group.

Each --run is "group|label|log_dir" where log_dir is the --log_dir of the training run (the stdout
log is expected at "<log_dir>.txt"). Lines in a group use one hue, light -> dark, in the given order.

Example:
    python scripts/plot_training_curves.py --out docs/figures/center_loss_es_curves.png \
        --run "latent 2|λ = 0|logs/center_es/ld2_lam0" --run "latent 2|λ = 1|logs/center_es/ld2_lam1"
"""

import os
import re
from argparse import ArgumentParser

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b", "#104281"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help='"group|label|log_dir" (repeat)')
    parser.add_argument("--title", default="Training curves")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_run(log_dir):
    text = open(f"{log_dir}.txt", encoding="utf-8", errors="ignore").read()
    aurocs = [float(v) for v in re.findall(r"AUROC: ([0-9.]+)", text)]
    val = [float(v) for v in re.findall(r"val loss \(held-out normal train data\): ([0-9.]+)", text)]
    best = int(re.search(r"Best epoch (\d+)", text).group(1))
    return aurocs, val, best


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main(args):
    groups = {}
    for spec in args.run:
        group, label, log_dir = spec.split("|")
        groups.setdefault(group, []).append((label, log_dir))

    fig, axes = plt.subplots(len(groups), 2, figsize=(14, 4.6 * len(groups)), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for row, (group, runs) in enumerate(groups.items()):
        ax_val, ax_auc = axes[row]
        for k, (label, log_dir) in enumerate(runs):
            aurocs, val, best = read_run(log_dir)
            epochs = range(1, len(aurocs) + 1)
            ax_val.plot(range(1, len(val) + 1), val, color=RAMP[k], linewidth=2, label=label)
            ax_auc.plot(epochs, aurocs, color=RAMP[k], linewidth=2, label=f"{label} (best-val epoch {best}: {aurocs[best - 1]:.3f})")
            ax_auc.scatter([best], [aurocs[best - 1]], color=RAMP[k], s=40, edgecolor="white", linewidth=1.5, zorder=5)
        ax_val.set_yscale("log")
        ax_val.set_title(f"{group}: validation loss (held-out normal train data)", fontsize=11, color=INK, loc="left")
        ax_val.set_xlabel("epoch", color=MUTED); ax_val.set_ylabel("training objective (log scale)", color=MUTED)
        ax_auc.axvline(15, color=MUTED, linestyle=":", linewidth=1)
        ax_auc.text(15.5, ax_auc.get_ylim()[0], "epoch 15", color=MUTED, fontsize=8, va="bottom")
        ax_auc.set_title(f"{group}: test AUROC (dot = epoch chosen by validation loss)", fontsize=11, color=INK, loc="left")
        ax_auc.set_xlabel("epoch", color=MUTED); ax_auc.set_ylabel("AUROC", color=MUTED)
        for ax in (ax_val, ax_auc):
            style(ax)
            ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(args.title, fontsize=14, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main(get_args())
