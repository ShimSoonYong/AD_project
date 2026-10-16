import os
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import auc, roc_curve
from data import SELECT_NORMAL


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
        os.makedirs(f"{self.LOG_DIR}/compare_io", mode=751, exist_ok=True)
        os.makedirs(f"{self.LOG_DIR}/hist", mode=751, exist_ok=True)
        os.makedirs(f"{self.LOG_DIR}/auroc", mode=751, exist_ok=True)
        os.makedirs(f"{self.LOG_DIR}/record", mode=751, exist_ok=True)

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

        plt.savefig(f"{self.LOG_DIR}/compare_io/epoch{epoch}_compare_io.png")
        plt.close()

    @torch.inference_mode()
    def collect_record(self, epoch: int):
        record = {"id": [], "label": [], "score": [], "normal": []}

        for step, (x, label) in enumerate(self.loader):
            x = x.to(self.DEVICE)
            decoded = self.model(x)
            loss = float(self.criterion(decoded, x).cpu().numpy())

            record["id"].append(step)
            record["label"].append(int(label != SELECT_NORMAL))
            record["score"].append(loss)  # anomaly score

            if loss > self.THRESHOLDVAL:
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
        plt.savefig(f"{self.LOG_DIR}/hist/epoch{epoch}_score_hist.png")
        plt.close()

    def plot_auc(self, epoch: int):
        # Generating AUROC
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html

        fpr, tpr, thresholds = roc_curve(
            self.record["label"], self.record["score"], pos_label=1
        )
        plt.plot(fpr, tpr)
        auroc = auc(fpr, tpr)
        print(f"AUROC: {auroc}")
        plt.title(f"AUROC: {auroc}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.savefig(f"{self.LOG_DIR}/auroc/epoch{epoch}_auroc.png")
        plt.close()

    def save_record(self, epoch: int):
        self.record.pop("normal", None)
        self.record.pop("label", None)
        df = pd.DataFrame.from_dict(self.record)
        teamnumber = 2  # insert your teamnumber
        df.to_csv(
            f"{self.LOG_DIR}/record/epoch{epoch}_result_team_{teamnumber}.csv",
            index=False,
        )  # Index should be not included in the .csv file.
