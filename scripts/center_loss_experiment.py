"""Experiment: autoencoder + latent center loss (Deep SVDD / DASVDD style).

This is the script that produced the results in docs/center_loss.md.

loss = MSE(x_hat, x) + lam * ||z - c||^2, c = mean train latent (recomputed each epoch, no grad)
Tutorial settings: MLP AE (784-128-latent-128-784), 5 epochs, batch 1, Adam lr 0.005, seed 0.
Everything fitted uses train (label 2) only; test labels are used only for AUROC.

Usage: python scripts/center_loss_experiment.py <latent_dim> <lam> [project_root]
Note: with lam=1 the latent collapses (std -> 0) and the "recon+dist" score divides by zero,
so the run stops with "Input contains NaN" (reported as collapse in the results).
"""

import os
import sys
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

latent_dim, lam = int(sys.argv[1]), float(sys.argv[2])
root = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{root}/src")
os.chdir(root)
from data import load_dataset, SELECT_NORMAL
from models import Autoencoder

torch.set_num_threads(1)
random.seed(0); np.random.seed(0); torch.manual_seed(0)

train_set, test_set = load_dataset(root="dataset")
x_tr = train_set.data.float().div(255).view(-1, 784)  # same as ToTensor
x_te = test_set.data.float().div(255).view(-1, 784)
y_te = test_set.targets.numpy()
is_abn = (y_te != SELECT_NORMAL).astype(int)

model = Autoencoder(latent_dim=latent_dim)
opt = torch.optim.Adam(model.parameters(), lr=0.005)


@torch.inference_mode()
def scores(x, c):
    model.eval()
    z = model.encoder(x)
    err = ((model.decoder(z) - x) ** 2).mean(1)
    dist = ((z - c) ** 2).sum(1)
    return err.numpy(), dist.numpy(), z


for epoch in range(1, 6):
    with torch.inference_mode():
        c = model.encoder(x_tr).mean(0)
    model.train()
    for i in torch.randperm(len(x_tr)):
        x = x_tr[i : i + 1]
        z = model.encoder(x)
        loss = ((model.decoder(z) - x) ** 2).mean() + lam * ((z - c) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.inference_mode():
        c = model.encoder(x_tr).mean(0)
    err_tr, dist_tr, z_tr = scores(x_tr, c)
    err_te, dist_te, _ = scores(x_te, c)
    comb = (err_te - err_tr.mean()) / err_tr.std() + (dist_te - dist_tr.mean()) / dist_tr.std()
    spread = z_tr.std(0).mean().item()
    print(
        f"ld{latent_dim} lam{lam} ep{epoch} | recon {roc_auc_score(is_abn, err_te):.4f}"
        f" | dist {roc_auc_score(is_abn, dist_te):.4f} | recon+dist {roc_auc_score(is_abn, comb):.4f}"
        f" | latent std {spread:.3f} | train mse {err_tr.mean():.4f}",
        flush=True,
    )
