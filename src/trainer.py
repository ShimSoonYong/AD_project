import torch
from tqdm import tqdm
from utils import Scorer
from torch import nn, optim
from torch.amp import autocast
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

    @autocast(device_type="cuda")
    def train(self, epoch: int):
        self.model.train()
        print(f"{epoch} epoch Training starts")

        loss = torch.tensor(0)
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
            pbar.set_postfix({"Loss": loss.item()})

    @torch.inference_mode()
    def validate(self, epoch: int):
        self.model.eval()
        print(f"{epoch} epoch Validation starts")

        loss = torch.tensor(0)
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

            pbar.set_postfix({"Loss": loss.item()})

    def fit(self):
        print(f"Model fitting for {self.epochs}")

        pbar = tqdm(range(1, self.epochs + 1), desc="Fitting:", leave=False)
        for epoch in pbar:
            self.train(epoch)
            self.validate(epoch)
        print("Fitting complete")

        return self.model
