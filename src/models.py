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

    def __init__(self, latent_dim=16):
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


class MemoryModule(nn.Module):
    """Map [batch, latent_dim] queries to memory vectors and attention weights.

    Sparse addressing follows https://arxiv.org/abs/1904.02639, Section 3.3.
    """

    def __init__(self, latent_dim, memory_size=100, shrink_threshold=None):
        super().__init__()

        if latent_dim < 1 or memory_size < 1:
            raise ValueError("latent_dim and memory_size must be positive")
        if shrink_threshold is None:
            shrink_threshold = 1.0 / memory_size
        if not 0 <= shrink_threshold <= 1:
            raise ValueError("shrink_threshold must be between 0 and 1")

        self.shrink_threshold = shrink_threshold
        self.weight = nn.Parameter(torch.empty(memory_size, latent_dim))
        bound = latent_dim ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, z):
        similarity = F.normalize(z, dim=1) @ F.normalize(self.weight, dim=1).T
        attention = F.softmax(similarity, dim=1)

        if self.shrink_threshold > 0:
            difference = attention - self.shrink_threshold
            sparse_attention = F.relu(difference) * attention / (difference.abs() + 1e-12)
            weight_sum = sparse_attention.sum(dim=1, keepdim=True)
            # Keep the soft attention if shrinkage removes every item in a row.
            attention = torch.where(
                weight_sum > 0,
                sparse_attention / weight_sum.clamp_min(1e-12),
                attention,
            )

        retrieved = attention @ self.weight
        return retrieved, attention


class MemAE(nn.Module):
    """Reconstruct [batch, 1, 28, 28] images using retrieved memory vectors."""

    def __init__(self, latent_dim=16, memory_size=100, shrink_threshold=None):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
        )

        self.memory = MemoryModule(latent_dim, memory_size, shrink_threshold)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, return_attention=False):
        encoded = self.encoder(x)
        retrieved, attention = self.memory(encoded)
        decoded = self.decoder(retrieved)
        if return_attention:
            return decoded, attention
        return decoded


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
