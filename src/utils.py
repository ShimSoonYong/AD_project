import os
import math
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import auc, roc_curve

ROOT_DIR = "/workspace/dataset/"

SELECT_NORMAL = 2  # Set 2 class as normal samples.


class Scorer:
    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        threshold: float,
        criterion: nn.Module,
        log_dir: str,
        device: torch.device,
    ):
        self.model = model
        self.loader = loader
        self.THRESHOLDVAL = threshold  # threshold val
        self.criterion = criterion
        self.LOG_DIR = log_dir
        self.DEVICE = device

        os.makedirs(f"{self.LOG_DIR}/compare_io", mode=755, exist_ok=True)
        os.makedirs(f"{self.LOG_DIR}/hist", mode=755, exist_ok=True)
        os.makedirs(f"{self.LOG_DIR}/auroc", mode=755, exist_ok=True)
        os.makedirs(f"{self.LOG_DIR}/record", mode=755, exist_ok=True)
        os.makedirs(f"{self.LOG_DIR}/metrics", mode=755, exist_ok=True)

        # Container for progressive epoch metrics
        self.history = {
            "epoch": [],
            "loss": [],
            "normal_score": [],
            "abnormal_score": [],
            "overall_score": [],
            "auroc": [],
        }

    @torch.inference_mode()
    def visualize_IO(self, epoch: int, samples: torch.Tensor, num_samples: int = 4):

        num_samples = min(num_samples, samples.size(0))
        samples = samples[:num_samples]

        was_training = self.model.training
        self.model.eval()
        try:
            decoded = self.model(samples)
        finally:
            self.model.train(was_training)

        samples = samples.cpu().numpy()
        decoded = decoded.cpu().numpy()
        _, ax = plt.subplots(2, num_samples, figsize=(num_samples, 2), squeeze=False)

        for i in range(num_samples):
            img = samples[i].reshape(28, 28, -1)
            ax[0][i].imshow(img, cmap="gray")
            ax[0][i].set_xticks(())
            ax[0][i].set_yticks(())

            if i == 0:
                ax[0][i].set_ylabel("Input")

        for i in range(num_samples):
            img = decoded[i].reshape(28, 28, -1)
            ax[1][i].imshow(img, cmap="gray")
            ax[1][i].set_xticks(())
            ax[1][i].set_yticks(())

            if i == 0:
                ax[1][i].set_ylabel("Output")

        plt.savefig(f"{self.LOG_DIR}/compare_io/epoch{epoch}_compare_io.png", dpi=500)
        plt.close()

    @torch.inference_mode()
    def collect_record(self):
        record = {"id": [], "label": [], "score": [], "normal": []}

        for step, (x, label) in enumerate(self.loader):
            x = x.to(self.DEVICE)
            decoded = self.model(x)
            loss = float(self.criterion(decoded, x).cpu().numpy())

            record["id"].append(step)
            record["label"].append(int(label != SELECT_NORMAL))
            record["score"].append(loss)

        self.THRESHOLDVAL = float(np.quantile(record["score"][record["label"] == 0], 0.95))

        for score in record["score"]:
            if score > self.THRESHOLDVAL:
                record["normal"].append("0")
            else:
                record["normal"].append("1")

        self.record = record

    def plot_score(self, epoch: int):
        label = np.array(self.record["label"])
        score = np.array(self.record["score"])

        score_min = score.min()
        score_max = score.max()

        plt.hist(
            score[np.where(label == 0)[0]],
            bins=30,
            range=(score_min, score_max),
            alpha=0.5,
            label="Normal",
        )
        plt.hist(
            score[np.where(label == 1)[0]],
            bins=30,
            range=(score_min, score_max),
            alpha=0.5,
            label="Abnormal",
        )
        plt.xlabel("Anomaly score")
        plt.ylabel("Frequency")
        plt.axvline(self.THRESHOLDVAL, 0, 1, color="red", linestyle="--", linewidth=1)
        plt.legend(loc="upper right")
        plt.savefig(f"{self.LOG_DIR}/hist/epoch{epoch}_score_hist.png", dpi=500)
        plt.close()

    def plot_auc(self, epoch: int):
        fpr, tpr, thresholds = roc_curve(
            self.record["label"], self.record["score"], pos_label=1
        )
        plt.plot(fpr, tpr)
        auroc = auc(fpr, tpr)
        print(f"AUROC: {auroc}")
        plt.title(f"AUROC: {auroc:.4f}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.savefig(f"{self.LOG_DIR}/auroc/epoch{epoch}_auroc.png", dpi=500)
        plt.close()

    def plot_metrics(self, epoch: int):
        """Calculates current epoch summary metrics, tracks history, and plots progressive curves."""
        scores = np.array(self.record["score"])
        labels = np.array(self.record["label"])

        mean_loss = float(np.mean(scores))
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        current_auroc = float(auc(fpr, tpr))

        normal_mask = labels == 0
        abnormal_mask = labels == 1
        mean_normal = (
            float(np.mean(scores[normal_mask])) if np.any(normal_mask) else 0.0
        )
        mean_abnormal = (
            float(np.mean(scores[abnormal_mask])) if np.any(abnormal_mask) else 0.0
        )

        self.history["epoch"].append(epoch)
        self.history["loss"].append(mean_loss)
        self.history["overall_score"].append(mean_loss)
        self.history["normal_score"].append(mean_normal)
        self.history["abnormal_score"].append(mean_abnormal)
        self.history["auroc"].append(current_auroc)

        epochs = self.history["epoch"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(
            epochs,
            self.history["loss"],
            marker="o",
            color="tab:blue",
            label="Recon Loss",
        )
        axes[0].set_title("Reconstruction Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend()

        axes[1].plot(
            epochs,
            self.history["normal_score"],
            marker="o",
            color="tab:green",
            label="Mean Normal",
        )
        axes[1].plot(
            epochs,
            self.history["abnormal_score"],
            marker="s",
            color="tab:red",
            label="Mean Abnormal",
        )
        axes[1].axhline(
            self.THRESHOLDVAL,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Threshold",
        )
        axes[1].set_title("Mean Anomaly Score Progression")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Anomaly Score")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend()

        axes[2].plot(
            epochs, self.history["auroc"], marker="^", color="tab:purple", label="AUROC"
        )
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_title("AUROC Progression")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("AUROC")
        axes[2].grid(True, linestyle="--", alpha=0.6)
        axes[2].legend()

        plt.tight_layout()

        plt.savefig(f"{self.LOG_DIR}/metrics/epoch{epoch}_metrics.png", dpi=500)
        plt.savefig(f"{self.LOG_DIR}/metrics/metrics_curves.png", dpi=500)
        plt.close()

    def save_record(self, epoch: int):
        rec_copy = self.record.copy()
        rec_copy.pop("normal", None)
        rec_copy.pop("label", None)
        df = pd.DataFrame.from_dict(rec_copy)
        teamnumber = 2
        df.to_csv(
            f"{self.LOG_DIR}/record/epoch{epoch}_result_team_{teamnumber}.csv",
            index=False,
        )


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int = 42):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


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
            F.conv2d(decoded * target, window, groups=channels)
            - mean_decoded * mean_target
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


class TVLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, x):
        batch_size, c, h, w = x.size()

        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()

        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()

        loss = self.loss_weight * (tv_h + tv_w) / (batch_size * c * h * w)
        return loss
