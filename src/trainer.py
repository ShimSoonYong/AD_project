import torch
from tqdm import tqdm
from torch import nn, optim
from utils import Scorer
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        epochs: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scorer: Scorer,
        device: torch.device,
    ):

        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scorer = scorer
        self.device = device

    def train(self, epoch: int):
        self.model.train()
        print(f"{epoch} epoch Training starts")

        pbar = tqdm(self.train_loader, desc=f"{epoch} Epoch training:", leave=False)
        for step, (x, label) in enumerate(pbar):
            x = x.to(self.device)

            decoded = self.model(x)

            loss = self.criterion(decoded, x)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pbar.set_postfix({"Loss:": loss.item()})
            self.scorer.visualize_IO(epoch, x)

    @torch.inference_mode()
    def validate(self, epoch: int):
        self.model.eval()
        print(f"{epoch} epoch Validation starts")

        pbar = tqdm(self.test_loader, desc=f"{epoch} Epoch validation:", leave=False)
        for step, (x, label) in enumerate(pbar):
            x = x.to(self.device)

            decoded = self.model(x)

            loss = self.criterion(decoded, x)
            pbar.set_postfix({"Loss": loss.item()})

        self.scorer.collect_record(epoch)
        self.scorer.plot_auc(epoch)
        self.scorer.plot_score(epoch)
        self.scorer.save_record(epoch)

    def fit(self):
        print(f"Model fitting for {self.epochs}")
        pbar = tqdm(range(1, self.epochs + 1), desc="Fitting:", leave=False)
        for epoch in pbar:
            self.train(epoch)
            self.validate(epoch)
        print("Fitting complete")

        return self.model
