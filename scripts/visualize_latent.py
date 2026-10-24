"""Visualize where the task's test labels (0, 2, 4, 6) land in an Autoencoder's latent space.

For each given model: (left) test latents in 2D -- the latent itself if it is 2-dimensional,
otherwise a PCA projection fitted on the TRAIN (normal) latents only; (right) the distribution
of the squared distance ||z - c||^2 per class, c = mean train latent, with the AUROC of that
distance used alone as an anomaly score. Nothing is fitted on test data.

Example:
    python scripts/visualize_latent.py --out docs/figures/latent_space.png \
        --model "latent 2:2:logs/latent/ld2/Autoencoder/checkpoint/epoch5_weight.pth" \
        --model "latent 8:8:logs/latent/ld8/Autoencoder/checkpoint/epoch5_weight.pth"
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

CLASSES = {0: "T-shirt", 2: "Pullover (normal)", 4: "Coat", 6: "Shirt"}
COLORS = {0: "#1f77b4", 2: "#d62728", 4: "#ff7f0e", 6: "#2ca02c"}


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help='"title:latent_dim:weight_path[:key=value,...]" (repeat for several models), '
        'e.g. extra "activation=leaky_relu,norm=layer"',
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--data_root", default=os.path.join(ROOT, "dataset"))
    parser.add_argument("--out", default=os.path.join(ROOT, "docs", "figures", "latent_space.png"))
    return parser.parse_args()


@torch.inference_mode()
def encode(weight, latent_dim, hidden_dim, extra, x_tr, x_te):
    model = Autoencoder(latent_dim=latent_dim, hidden_dim=hidden_dim, **extra)
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    model.eval()
    # Hidden units that give a positive output for at least one test image ("alive")
    hidden = model.encoder[:-1](x_te.view(-1, 28 * 28))
    alive = int((hidden > 0).any(0).sum())
    return model.encode(x_tr).numpy(), model.encode(x_te).numpy(), alive, hidden.shape[1]


def main(args):
    train_set, test_set = load_dataset(root=args.data_root)
    x_tr = train_set.data.float().div(255).view(-1, 1, 28, 28)  # same as ToTensor
    x_te = test_set.data.float().div(255).view(-1, 1, 28, 28)
    y_te = test_set.targets.numpy()
    is_abnormal = (y_te != SELECT_NORMAL).astype(int)

    fig, axes = plt.subplots(len(args.model), 2, figsize=(12, 5 * len(args.model)), squeeze=False)
    for row, spec in enumerate(args.model):
        title, latent_dim, weight, *rest = spec.split(":", 3)
        extra = dict(kv.split("=") for kv in rest[0].split(",")) if rest else {}
        z_tr, z_te, alive, hidden = encode(weight, int(latent_dim), args.hidden_dim, extra, x_tr, x_te)
        title += f" [alive units {alive}/{hidden}, latent std {z_tr.std(0).mean():.3f}]"
        center = z_tr.mean(0)
        dist = ((z_te - center) ** 2).sum(1)

        # 2D view: the latent itself, or PCA fitted on normal TRAIN latents only
        if z_tr.shape[1] == 2:
            p_te, p_c, axis_name = z_te, center, "z"
        else:
            pca = PCA(n_components=2).fit(z_tr)
            p_te, p_c = pca.transform(z_te), pca.transform(center[None])[0]
            ratio = np.nan_to_num(pca.explained_variance_ratio_.sum())  # NaN if the latent collapsed
            axis_name = "PC"
            title += f"\n(PCA of train latents, {ratio:.0%} of variance shown)"

        ax = axes[row][0]
        for c in (0, 4, 6, 2):  # draw normal last so it stays visible
            m = y_te == c
            ax.scatter(p_te[m, 0], p_te[m, 1], s=4, alpha=0.45, color=COLORS[c], label=CLASSES[c])
        ax.scatter(*p_c, marker="*", s=400, color="black", edgecolor="white", label="c (mean train latent)")
        ax.set_title(title)
        ax.set_xlabel(f"{axis_name}1"); ax.set_ylabel(f"{axis_name}2")
        legend = ax.legend(fontsize=8, loc="best")
        for handle in legend.legend_handles:
            handle.set_sizes([30])  # same legend marker size for points and the center star

        ax = axes[row][1]
        clip = max(np.percentile(dist, 99), 1e-12)  # a collapsed latent gives all-zero distances
        bins = np.linspace(0, clip, 50)
        for c in (2, 0, 4, 6):
            m = y_te == c
            ax.hist(np.clip(dist[m], 0, clip), bins=bins, histtype="step", linewidth=1.8,
                    color=COLORS[c], label=f"{CLASSES[c]} (median {np.median(dist[m]):.2f})")
        auroc = roc_auc_score(is_abnormal, dist)
        ax.set_title(f"{title.split(' [')[0].split(' (PCA')[0]}\n||z - c||^2 per class | AUROC of distance alone = {auroc:.3f}")
        ax.set_xlabel("||z - c||^2 (clipped at 99th percentile)"); ax.set_ylabel("count")
        ax.legend(fontsize=8)
        print(f"{title}: distance AUROC {auroc:.4f}; median distance "
              + ", ".join(f"{CLASSES[c]} {np.median(dist[y_te == c]):.3f}" for c in (2, 0, 4, 6)))

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main(get_args())
