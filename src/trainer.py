import csv
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils import Scorer
from models import FeatureVAE, MemAE
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader


class SSIMLoss(nn.Module):
    """Return 1 - mean SSIM for [0, 1] images using a 7x7 Gaussian window."""

    def __init__(self):
        super().__init__()
        coordinates = torch.arange(7, dtype=torch.float32) - 3
        gaussian = torch.exp(-coordinates.square() / (2 * 1.5**2))
        gaussian = gaussian / gaussian.sum()
        self.register_buffer("window", torch.outer(gaussian, gaussian)[None, None])

    def forward(self, decoded: torch.Tensor, target: torch.Tensor):
        channels = decoded.size(1)
        window = self.window.to(decoded).expand(channels, 1, -1, -1)
        mean_decoded = F.conv2d(decoded, window, groups=channels)
        mean_target = F.conv2d(target, window, groups=channels)
        variance_decoded = (
            F.conv2d(decoded.square(), window, groups=channels) - mean_decoded.square()
        )
        variance_target = (
            F.conv2d(target.square(), window, groups=channels) - mean_target.square()
        )
        covariance = (
            F.conv2d(decoded * target, window, groups=channels) - mean_decoded * mean_target
        )

        c1, c2 = 0.01**2, 0.03**2  # Input data range is 1.
        numerator = (2 * mean_decoded * mean_target + c1) * (2 * covariance + c2)
        denominator = (mean_decoded.square() + mean_target.square() + c1) * (
            variance_decoded + variance_target + c2
        )
        return 1 - (numerator / denominator).mean()


def memory_auxiliary_losses(encoded, memory_weight, margin=0.2):
    """Pull normalized queries toward their nearest detached memory prototype."""
    if memory_weight.size(0) < 2:
        raise ValueError("Memory auxiliary losses require at least two memory vectors")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("memory_margin must be positive and finite")

    queries = F.normalize(encoded, dim=1)
    prototypes = F.normalize(memory_weight.detach(), dim=1)
    nearest = (queries @ prototypes.T).topk(2, dim=1).indices
    positive, negative = prototypes[nearest[:, 0]], prototypes[nearest[:, 1]]
    compactness = F.mse_loss(queries, positive)
    separateness = F.triplet_margin_loss(
        queries, positive, negative, margin=margin, p=2, eps=1e-6, reduction="mean"
    )
    return compactness, separateness


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scorer: Scorer,
        device: torch.device,
        entropy_weight: float = 0.0001,
        compactness_weight: float = 0.0,
        separateness_weight: float = 0.0,
        memory_margin: float = 0.2,
    ):

        for name, weight in (
            ("entropy_weight", entropy_weight),
            ("compactness_weight", compactness_weight),
            ("separateness_weight", separateness_weight),
        ):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        if not math.isfinite(memory_margin) or memory_margin <= 0:
            raise ValueError("memory_margin must be positive and finite")
        if compactness_weight > 0 or separateness_weight > 0:
            if not isinstance(model, MemAE):
                raise ValueError("Memory auxiliary losses require a MemAE model")
            if model.memory.weight.size(0) < 2:
                raise ValueError("Memory auxiliary losses require at least two memory vectors")

        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scorer = scorer
        self.device = device
        self.entropy_weight = entropy_weight
        self.compactness_weight = compactness_weight
        self.separateness_weight = separateness_weight
        self.memory_margin = memory_margin

    def _compute_loss(self, x):
        """Use the same objective for training and validation."""
        if self.compactness_weight > 0 or self.separateness_weight > 0:
            decoded, attention, encoded = self.model(x, return_features=True)
            compactness, separateness = memory_auxiliary_losses(
                encoded, self.model.memory.weight, self.memory_margin
            )
            loss = self.criterion(decoded, x)
            if self.entropy_weight > 0:
                entropy_loss = -(attention * attention.clamp_min(1e-12).log()).sum(dim=1).mean()
                loss = loss + self.entropy_weight * entropy_loss
            return (loss + self.compactness_weight * compactness
                    + self.separateness_weight * separateness)

        if isinstance(self.model, MemAE) and self.entropy_weight > 0:
            decoded, attention = self.model(x, return_attention=True)
            # Sum over memory slots, then average over the batch; avoid log(0).
            entropy_loss = -(attention * attention.clamp_min(1e-12).log()).sum(dim=1).mean()
            return self.criterion(decoded, x) + self.entropy_weight * entropy_loss

        return self.criterion(self.model(x), x)

    def train(self, epoch: int):
        self.model.train()
        print(f"{epoch} epoch Training starts")

        loss = torch.tensor(0)
        total_loss = 0.0
        total_samples = 0
        pbar = tqdm(
            self.train_loader,
            desc=f"{epoch} Epoch training:",
            postfix={"Loss": loss.item()},
            leave=False,
        )
        for step, (x, label) in enumerate(pbar):
            x = x.to(self.device)

            loss = self._compute_loss(x)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_loss = loss.item()
            total_loss += batch_loss * x.size(0)
            total_samples += x.size(0)
            pbar.set_postfix({"Loss": batch_loss})

        if epoch == self.epochs and total_samples > 0:
            self.scorer.visualize_IO(epoch, x)

        return total_loss / total_samples

    @torch.inference_mode()
    def validate(self, epoch: int):
        self.model.eval()
        print(f"{epoch} epoch Validation starts")

        loss = torch.tensor(0)
        total_loss = 0.0
        total_samples = 0
        pbar = tqdm(
            self.val_loader,
            desc=f"{epoch} Epoch validation:",
            postfix={"Loss": loss.item()},
            leave=False,
        )
        for step, (x, label) in enumerate(pbar):
            x = x.to(self.device)

            loss = self._compute_loss(x)

            batch_loss = loss.item()
            total_loss += batch_loss * x.size(0)
            total_samples += x.size(0)
            pbar.set_postfix({"Loss": batch_loss})

        return total_loss / total_samples

    def fit(self):
        print(f"Model fitting for {self.epochs}")

        loss_history = []
        try:
            pbar = tqdm(range(1, self.epochs + 1), desc="Fitting:", leave=False)
            for epoch in pbar:
                train_loss = self.train(epoch)
                val_loss = self.validate(epoch)
                loss_history.append([epoch, train_loss, val_loss])
                tqdm.write(
                    f"Epoch {epoch}/{self.epochs} - "
                    f"train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}"
                )
        finally:
            # Save completed epochs once, including when training is interrupted.
            with open(
                f"{self.scorer.LOG_DIR}/loss_history.csv",
                "w",
                newline="",
                encoding="utf-8",
            ) as loss_file:
                writer = csv.writer(loss_file)
                writer.writerow(["epoch", "train_loss", "val_loss"])
                writer.writerows(loss_history)

        print("Fitting complete")

        return self.model


class FeatureTrainer:
    """Train cached feature reconstruction with run126's validation selection."""

    def __init__(
        self, model, epochs, features, optimizer, scorer, device,
        batch_size=64, entropy_weight=1.3e-6, seed=42, patience=80, kl_weight=1e-5,
    ):
        if epochs < 1 or batch_size < 1 or patience < 0:
            raise ValueError("Require epochs > 0, batch_size > 0, patience >= 0")
        if not math.isfinite(entropy_weight) or entropy_weight < 0:
            raise ValueError("entropy_weight must be non-negative and finite")
        if not math.isfinite(kl_weight) or kl_weight < 0:
            raise ValueError("kl_weight must be non-negative and finite")
        if features["normal_val_count"] != 600:
            raise ValueError("Feature validation must begin with 600 normal samples")
        if not torch.all(features["tuning_labels"][:600] == 2):
            raise ValueError("The first 600 validation labels must be class 2")

        self.model, self.optimizer, self.scorer = model, optimizer, scorer
        self.device = torch.device(device)
        self.epochs, self.batch_size = epochs, batch_size
        self.entropy_weight, self.seed, self.patience = entropy_weight, seed, patience
        self.kl_weight = kl_weight
        self.is_vae = isinstance(model, FeatureVAE)
        self.regularizer_name = "kl" if self.is_vae else "entropy"
        checkpoint_model = "FeatureVAE" if self.is_vae else "MemAE"
        self.best_checkpoint = f"best_{checkpoint_model}_weight.pth"
        self.final_checkpoint = f"final_{checkpoint_model}_weight.pth"
        self.train_x = features["train_x"].to(self.device)
        self.tuning_x = features["tuning_x"].to(self.device)
        self.tuning_labels = features["tuning_labels"].cpu().numpy()
        self.best_epoch = self.completed_epochs = 0
        self.selected_validation = self.tuning_scores = self.final_validation = None
        self.history = []

    def _loss_terms(self, batch):
        if self.is_vae:
            decoded, mu, logvar = self.model(batch, return_distribution=True)
            reconstruction = (decoded - batch).square().mean()
            # Sum over latent dimensions, then average over samples.
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
            return torch.stack((reconstruction + self.kl_weight * kl, reconstruction, kl))
        decoded, attention = self.model(batch, return_attention=True)
        reconstruction = (decoded - batch).square().mean()
        entropy = -(attention * attention.clamp_min(1e-12).log()).sum(1).mean()
        return torch.stack((
            reconstruction + self.entropy_weight * entropy, reconstruction, entropy,
        ))

    @torch.inference_mode()
    def evaluate(self, features, labels, batch_size=256):
        self.model.eval()
        features = features.to(self.device)
        labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
        values = [(self.model(batch) - batch).square().mean(1)
                  for batch in features.split(batch_size)]
        scores = torch.cat(values).cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite feature reconstruction scores")
        metrics = {"auroc": float(roc_auc_score(labels != 2, scores))}
        for label in (0, 4, 6):
            selected = np.isin(labels, (2, label))
            metrics[f"class_{label}_vs_2"] = float(
                roc_auc_score(labels[selected] != 2, scores[selected])
            )
        return metrics, scores

    @torch.inference_mode()
    def _normal_validation(self):
        self.model.eval()
        total = torch.zeros(3, device=self.device)
        for batch in self.tuning_x[:600].split(self.batch_size):
            total += self._loss_terms(batch) * len(batch)
        return (total / 600).tolist()

    def fit(self):
        run = Path(self.scorer.LOG_DIR)
        generator = torch.Generator().manual_seed(self.seed)
        best_auc, best_state = -1.0, None
        self.history = []
        try:
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                total = torch.zeros(3, device=self.device)
                permutation = torch.randperm(len(self.train_x), generator=generator).to(self.device)
                for indices in permutation.split(self.batch_size):
                    batch = self.train_x[indices]
                    self.optimizer.zero_grad(set_to_none=True)
                    terms = self._loss_terms(batch)
                    terms[0].backward()
                    self.optimizer.step()
                    total += terms.detach() * len(batch)
                self.completed_epochs = epoch
                training = (total / len(self.train_x)).tolist()
                validation_loss = self._normal_validation()
                if not np.isfinite(training + validation_loss).all():
                    raise ValueError("Non-finite feature training or validation loss")
                validation, _ = self.evaluate(self.tuning_x, self.tuning_labels)
                if validation["auroc"] > best_auc:
                    best_auc, self.best_epoch = validation["auroc"], epoch
                    best_state = {key: value.detach().cpu().clone()
                                  for key, value in self.model.state_dict().items()}
                    torch.save(best_state, run / self.best_checkpoint)
                self.history.append({
                    "epoch": epoch, "train_loss": training[0], "val_loss": validation_loss[0],
                    "tuning_auroc": validation["auroc"],
                    "train_reconstruction": training[1], "val_reconstruction": validation_loss[1],
                    f"train_{self.regularizer_name}": training[2],
                    f"val_{self.regularizer_name}": validation_loss[2],
                })
                tqdm.write(f"Epoch {epoch}/{self.epochs} - train_loss: {training[0]:.8f}, "
                           f"val_loss: {validation_loss[0]:.8f}, tuning_auroc: {validation['auroc']:.6f}")
                if epoch >= 100 and epoch - self.best_epoch >= self.patience:
                    break
            if best_state is None:
                raise RuntimeError("No feature checkpoint was selected")
            torch.save(self.model.state_dict(), run / self.final_checkpoint)
            self.final_validation, _ = self.evaluate(self.tuning_x, self.tuning_labels)
            self.model.load_state_dict(best_state)
            self.selected_validation, self.tuning_scores = self.evaluate(self.tuning_x, self.tuning_labels)
        finally:
            with (run / "loss_history.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "epoch", "train_loss", "val_loss", "tuning_auroc",
                    "train_reconstruction", "val_reconstruction",
                    f"train_{self.regularizer_name}", f"val_{self.regularizer_name}",
                ])
                writer.writeheader()
                writer.writerows(self.history)
        return self.model
