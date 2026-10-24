"""ROC curves of the reconstruction-error score for several saved Autoencoders.

Each --model is "panel title|label|latent_dim|weight_path"; models with the same panel title
share one panel. Colors follow the order of the models in a panel (one hue, light -> dark),
so pass them in increasing order of the varied parameter (e.g. λ). Test labels are used only
for the ROC curve.

Example:
    python scripts/plot_roc.py --out docs/figures/center_loss_bs64_roc.png \
        --model "latent 2|λ = 0|2|logs/deadrelu/gentle/Autoencoder/checkpoint/epoch15_weight.pth" \
        --model "latent 2|λ = 0.01|2|logs/center/bs64_ld2_lam0.01/Autoencoder/checkpoint/epoch15_weight.pth"
"""

import os
import sys
from argparse import ArgumentParser

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import load_dataset, SELECT_NORMAL
from models import Autoencoder

# Sequential blue ramp (reference palette steps 250 -> 700), light = small parameter
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b", "#104281"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", action="append", required=True,
                        help='"panel title|label|latent_dim|weight_path" (repeat)')
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--title", default="ROC curves (score = reconstruction error)")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


@torch.inference_mode()
def scores(weight, latent_dim, hidden_dim, x):
    model = Autoencoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    model.eval()
    return ((model(x) - x) ** 2).flatten(1).mean(1).numpy()


def main(args):
    _, test_set = load_dataset(root=os.path.join(ROOT, "dataset"))
    x = test_set.data.float().div(255).view(-1, 1, 28, 28)
    is_abnormal = (test_set.targets != SELECT_NORMAL).numpy()

    panels = {}
    for spec in args.model:
        panel, label, latent_dim, weight = spec.split("|")
        panels.setdefault(panel, []).append((label, int(latent_dim), weight))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (panel, models) in zip(axes[0], panels.items()):
        ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="random (0.500)")
        for k, (label, latent_dim, weight) in enumerate(models):
            fpr, tpr, _ = roc_curve(is_abnormal, scores(weight, latent_dim, args.hidden_dim, x))
            value = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=RAMP[k], linewidth=2, label=f"{label} (AUROC {value:.3f})")
            print(f"{panel} | {label}: AUROC {value:.4f}")
        ax.set_title(panel, fontsize=12, color=INK, loc="left")
        ax.set_xlabel("False positive rate", color=MUTED)
        ax.set_ylabel("True positive rate", color=MUTED)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], frameon=False, fontsize=9, loc="lower right")

    fig.suptitle(args.title, fontsize=14, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main(get_args())
