"""Side-by-side comparison of two Autoencoders (e.g. center loss λ=0 vs λ=0.01).

Top row: test latents of model A, of model B, and of model B drawn on model A's axis range
(shows how much the latent shrank). Bottom row: per-class reconstruction error of A and B
on a shared scale, and per-class AUROC of both. Uses test labels only for evaluation.

Example:
    python scripts/compare_center_loss.py --latent_dim 2 \
        --a "λ = 0:logs/deadrelu/gentle/Autoencoder/checkpoint/epoch15_weight.pth" \
        --b "λ = 0.01:logs/center/bs64_ld2_lam0.01/Autoencoder/checkpoint/epoch15_weight.pth" \
        --out docs/figures/center_loss_bs64_latent2_lam0_vs_0.01.png
"""

import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import load_dataset, SELECT_NORMAL
from models import Autoencoder

# Fixed categorical order (reference palette slots 1-4); identity also given by legend/labels
CLASSES = [(2, "Pullover (normal)", "#2a78d6"), (0, "T-shirt", "#eb6834"),
           (4, "Coat", "#1baf7a"), (6, "Shirt", "#eda100")]
MODEL_COLORS = ("#8a8984", "#6250d6")  # model A (baseline) gray, model B violet
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--a", required=True, help='"label:weight_path" (baseline)')
    parser.add_argument("--b", required=True, help='"label:weight_path"')
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--title", default="Center loss on the latent space (latent 2, batch 64, 15 epochs)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--latent_only", action="store_true", help="Only draw the two latent-space panels")
    return parser.parse_args()


@torch.inference_mode()
def evaluate(weight, args, x_tr, x_te, y):
    model = Autoencoder(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    model.eval()
    z_tr, z_te = model.encode(x_tr).numpy(), model.encode(x_te).numpy()
    std = z_tr.std(0).mean()
    pca_ratio = None
    if z_tr.shape[1] > 2:
        # Project to 2D with a PCA fitted on the normal TRAIN latents only
        pca = PCA(n_components=2).fit(z_tr)
        pca_ratio = pca.explained_variance_ratio_.sum()
        z_tr, z_te = pca.transform(z_tr), pca.transform(z_te)
    err_tr = ((model(x_tr) - x_tr) ** 2).flatten(1).mean(1).numpy()
    err = ((model(x_te) - x_te) ** 2).flatten(1).mean(1).numpy()
    auroc = {"All": roc_auc_score(y != SELECT_NORMAL, err)}
    for c, name, _ in CLASSES[1:]:
        m = (y == c) | (y == SELECT_NORMAL)
        auroc[name] = roc_auc_score(y[m] != SELECT_NORMAL, err[m])
    return dict(z=z_te, c=z_tr.mean(0), std=std, pca_ratio=pca_ratio, err=err, train_err=err_tr.mean(), auroc=auroc)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def scatter(ax, res, y, title, limits=None):
    for c, name, color in CLASSES[::-1]:  # normal drawn last, on top
        m = y == c
        ax.scatter(res["z"][m, 0], res["z"][m, 1], s=6, alpha=0.5, color=color, label=name, linewidths=0)
    ax.scatter(*res["c"][:2], marker="*", s=260, color=INK, edgecolor="white", linewidth=1.5, label="c (mean train latent)", zorder=5)
    if limits:
        ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    axis = "z" if res["pca_ratio"] is None else "PC"
    ax.set_xlabel(f"{axis}1", color=MUTED); ax.set_ylabel(f"{axis}2", color=MUTED)
    style(ax)


def error_box(ax, res, y, title, ylim):
    data = [res["err"][y == c] for c, _, _ in CLASSES]
    box = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                     medianprops=dict(color=INK, linewidth=1.5))
    for patch, (_, _, color) in zip(box["boxes"], CLASSES):
        patch.set_facecolor(color); patch.set_alpha(0.75); patch.set_edgecolor("white")
    for i, d in enumerate(data, start=1):
        ax.text(i + 0.3, np.median(d), f"{np.median(d):.3f}", va="center", ha="left", fontsize=8, color=INK)
    ratio = np.mean(data[2]) / np.mean(data[0])
    ax.set_xticks(range(1, 5), [n.split(" (")[0] for _, n, _ in CLASSES], fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_title(f"{title}\nmean Coat / Pullover error = {ratio:.2f}", fontsize=11, color=INK, loc="left")
    ax.set_ylabel("reconstruction error (MSE)", color=MUTED)
    style(ax)


def main(args):
    train_set, test_set = load_dataset(root=os.path.join(ROOT, "dataset"))
    x_tr = train_set.data.float().div(255).view(-1, 1, 28, 28)
    x_te = test_set.data.float().div(255).view(-1, 1, 28, 28)
    y = test_set.targets.numpy()
    (la, wa), (lb, wb) = (s.split(":", 1) for s in (args.a, args.b))
    a, b = evaluate(wa, args, x_tr, x_te, y), evaluate(wb, args, x_tr, x_te, y)

    def latent_title(label, res, extra=""):
        pca = "" if res["pca_ratio"] is None else f", PCA of train latents ({res['pca_ratio']:.0%} of variance)"
        return f"{label}: latent space{extra}\nlatent std {res['std']:.3f}{pca}"

    def add_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles[::-1], labels[::-1], fontsize=8, frameon=False, loc="upper left")
        for handle in legend.legend_handles:
            handle.set_sizes([40])  # same legend marker size for the points and the center star

    if args.latent_only:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
        fig.patch.set_facecolor("#fcfcfb")
        scatter(axes[0], a, y, latent_title(la, a))
        scatter(axes[1], b, y, latent_title(lb, b, " (own scale)"))
        add_legend(axes[0])
        fig.suptitle(args.title, fontsize=14, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        fig.savefig(args.out, dpi=130, facecolor=fig.get_facecolor())
        print("saved", args.out)
        for label, res in ((la, a), (lb, b)):
            print(label, "| std %.3f | PCA ratio %s" % (res["std"], res["pca_ratio"]))
        return

    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    fig.patch.set_facecolor("#fcfcfb")
    scatter(axes[0][0], a, y, latent_title(la, a))
    scatter(axes[0][1], b, y, latent_title(lb, b, " (own scale)"))
    scatter(axes[0][2], b, y, f"{lb} on the {la} axis range\n(same points as the middle panel)",
            limits=(axes[0][0].get_xlim(), axes[0][0].get_ylim()))
    add_legend(axes[0][0])

    top = max(np.percentile(r["err"][y == c], 92) for r in (a, b) for c, _, _ in CLASSES)
    error_box(axes[1][0], a, y, f"{la}: test error per class (train error {a['train_err']:.4f})", (0, top))
    error_box(axes[1][1], b, y, f"{lb}: test error per class (train error {b['train_err']:.4f})", (0, top))

    ax = axes[1][2]
    names = list(a["auroc"])
    pos = np.arange(len(names))
    for k, (res, label, color) in enumerate(((a, la, MODEL_COLORS[0]), (b, lb, MODEL_COLORS[1]))):
        vals = [res["auroc"][n] for n in names]
        bars = ax.bar(pos + (k - 0.5) * 0.36, vals, width=0.34, color=color, label=label, edgecolor="white", linewidth=2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(pos, ["All", "T-shirt\nvs Pullover", "Coat\nvs Pullover", "Shirt\nvs Pullover"], fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC (reconstruction error)", color=MUTED)
    ax.set_title("AUROC per class (epoch 15)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style(ax)

    fig.suptitle(args.title, fontsize=14, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor=fig.get_facecolor())
    print("saved", args.out)
    for label, res in ((la, a), (lb, b)):
        print(label, "| std %.3f | train err %.4f |" % (res["std"], res["train_err"]),
              " ".join(f"{k} {v:.3f}" for k, v in res["auroc"].items()))


if __name__ == "__main__":
    main(get_args())
