import torch
from torch import nn
import torch.nn.functional as F


# basic autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1, 1, 28, 28)

        return decoded


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, 2, 1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 4, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.ConvTranspose2d(4, 1, 3, 2, 1, 1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


class CAE(nn.Module):
    """Reconstruct [batch, 1, 28, 28] images through a compact latent vector."""

    def __init__(self, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == "__main__":
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        model = CAE().to(device).eval()
        with torch.inference_mode():
            for batch_size in (1, 32):
                inputs = torch.rand(batch_size, 1, 28, 28, device=device)
                outputs = model(inputs)

                assert outputs.shape == inputs.shape
                assert outputs.dtype == inputs.dtype
                assert outputs.device == inputs.device
                assert torch.isfinite(outputs).all().item()
                minimum = outputs.min().item()
                maximum = outputs.max().item()
                assert 0 <= minimum <= maximum <= 1

                print(
                    f"CAE [{device}] {tuple(inputs.shape)} -> {tuple(outputs.shape)} "
                    f"| {outputs.dtype} | range: [{minimum:.4f}, {maximum:.4f}] | PASS"
                )
