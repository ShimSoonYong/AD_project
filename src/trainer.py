import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import Scorer
from scoring import perceptual_distance
from torch import nn, optim
from torch.amp import autocast
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
        checkpoint_dir: str,
        mask_ratio: float = 0.0,
        perceptual_weight: float = 0.0,
        center_weight: float = 0.0,
        val_loader: DataLoader = None,
        patience: int = 0,
    ):

        self.model = model
        # torch.compile needs a C++ compiler on CPU and Triton on CUDA (not available on Windows),
        # so only compile on Linux + CUDA
        if device.type == "cuda" and sys.platform != "win32":
            self.model.compile()
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scorer = scorer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.mask_ratio = mask_ratio
        self.perceptual_weight = perceptual_weight
        self.center_weight = center_weight
        # Early stopping on the training objective of held-out NORMAL train data (never test data)
        self.val_loader = val_loader
        self.patience = patience

    @torch.no_grad()
    def latent_center(self):
        # c = mean latent of the (normal) train data under the current encoder, fixed for one epoch
        self.model.eval()
        latents = [self.model.encode(x.to(self.device)) for x, _ in self.train_loader]
        self.model.train()
        return torch.cat(latents).mean(0)

    def mask(self, x: torch.Tensor):
        # Training only: zero random 7x7 blocks (4x4 grid) of the input; the target stays the full image
        keep = torch.rand(len(x), 1, 4, 4, device=x.device) >= self.mask_ratio
        return x * F.interpolate(keep.float(), scale_factor=7, mode="nearest")

    def objective(self, x: torch.Tensor, x_in: torch.Tensor, center: torch.Tensor):
        # Training loss; also used on held-out normal data for early stopping
        if self.center_weight > 0:
            # loss = recon_error + center_weight * ||z - c||^2 (pulls normal latents to c)
            z = self.model.encode(x_in)
            decoded = self.model.decode(z)
            loss = self.criterion(decoded, x)
            loss = loss + self.center_weight * ((z - center) ** 2).sum(1).mean()
        else:
            decoded = self.model(x_in)
            loss = self.criterion(decoded, x)
        if self.perceptual_weight > 0:
            loss = loss + self.perceptual_weight * perceptual_distance(x, decoded).mean()
        return loss

    @torch.no_grad()
    def validation_loss(self):
        center = self.latent_center() if self.center_weight > 0 else None
        self.model.eval()
        total, count = 0.0, 0
        for x, _ in self.val_loader:
            x = x.to(self.device)
            total += self.objective(x, x, center).item() * len(x)
            count += len(x)
        return total / count

    def train(self, epoch: int):
        self.model.train()
        print(f"{epoch} epoch Training starts")
        center = self.latent_center() if self.center_weight > 0 else None

        loss = torch.tensor(0)
        pbar = tqdm(
            self.train_loader,
            desc=f"{epoch} Epoch training:",
            postfix={"Loss": loss.item()},
            leave=False,
        )
        for step, (x, label) in enumerate(pbar):
            x = x.to(self.device)
            x_in = self.mask(x) if self.mask_ratio > 0 else x

            with autocast(
                device_type=self.device.type, enabled=self.device.type == "cuda"
            ):
                loss = self.objective(x, x_in, center)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pbar.set_postfix({"Loss": loss.item()})

        # Save input/output comparison once per epoch (last batch) instead of every step
        self.scorer.visualize_IO(epoch, x)

    @torch.inference_mode()
    def validate(self, epoch: int):
        self.model.eval()
        print(f"{epoch} epoch Validation starts")

        loss = torch.tensor(0)
        pbar = tqdm(
            self.test_loader,
            desc=f"{epoch} Epoch validation:",
            postfix={"Loss": loss.item()},
            leave=False,
        )
        for step, (x, label) in enumerate(pbar):
            x = x.to(self.device)

            decoded = self.model(x)

            loss = self.criterion(decoded, x)

            pbar.set_postfix({"Loss": loss.item()})

        self.scorer.collect_record(epoch)
        self.scorer.plot_auc(epoch)
        self.scorer.plot_score(epoch)
        self.scorer.save_record(epoch)
        # Save the weight that produced this epoch's csv, so any submitted csv can be reproduced
        torch.save(
            self.model.state_dict(), f"{self.checkpoint_dir}/epoch{epoch}_weight.pth"
        )

    def fit(self):
        print(f"Model fitting for {self.epochs}")

        best_loss, best_epoch, history = float("inf"), 0, []
        pbar = tqdm(range(1, self.epochs + 1), desc="Fitting:", leave=False)
        for epoch in pbar:
            self.train(epoch)
            self.validate(epoch)
            if self.val_loader is None:
                continue
            val_loss = self.validation_loss()
            history.append(val_loss)
            print(f"{epoch} epoch val loss (held-out normal train data): {val_loss:.6f}")
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
            elif self.patience > 0 and epoch - best_epoch >= self.patience:
                print(f"Early stopping at epoch {epoch}: no improvement for {self.patience} epochs")
                break
        if self.val_loader is not None:
            # The epoch with the lowest val loss is chosen without any test label
            print(f"Best epoch {best_epoch} (val loss {best_loss:.6f})")
            with open(f"{self.checkpoint_dir}/early_stopping.txt", "w") as f:
                f.write(f"best_epoch {best_epoch}\nbest_val_loss {best_loss}\n")
                f.writelines(f"epoch{i} {v}\n" for i, v in enumerate(history, start=1))
        print("Fitting complete")

        return self.model
