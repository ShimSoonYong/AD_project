import csv

import torch
from tqdm import tqdm
from utils import Scorer
from torch import nn, optim
from torch.utils.data import DataLoader


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
    ):

        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scorer = scorer
        self.device = device

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

            decoded = self.model(x)

            loss = self.criterion(decoded, x)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scorer.visualize_IO(epoch, x)
            batch_loss = loss.item()
            total_loss += batch_loss * x.size(0)
            total_samples += x.size(0)
            pbar.set_postfix({"Loss": batch_loss})

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

            decoded = self.model(x)

            loss = self.criterion(decoded, x)

            batch_loss = loss.item()
            total_loss += batch_loss * x.size(0)
            total_samples += x.size(0)
            pbar.set_postfix({"Loss": batch_loss})

        return total_loss / total_samples

    def fit(self):
        print(f"Model fitting for {self.epochs}")

        with open(
            f"{self.scorer.LOG_DIR}/loss_history.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as loss_file:
            writer = csv.writer(loss_file)
            writer.writerow(["epoch", "train_loss", "val_loss"])

            pbar = tqdm(range(1, self.epochs + 1), desc="Fitting:", leave=False)
            for epoch in pbar:
                train_loss = self.train(epoch)
                val_loss = self.validate(epoch)
                writer.writerow([epoch, train_loss, val_loss])
                loss_file.flush()
                tqdm.write(
                    f"Epoch {epoch}/{self.epochs} - "
                    f"train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}"
                )
        print("Fitting complete")

        return self.model
