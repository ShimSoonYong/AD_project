import hashlib
import math
import os
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

class CAE(nn.Module):
    """Reconstruct [batch, 1, 28, 28] images through a compact latent vector."""

    def __init__(self, latent_dim=8):
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

    def __init__(self, latent_dim, memory_size, shrink_threshold=None, temperature=1.0):
        super().__init__()

        if latent_dim < 1 or memory_size < 1:
            raise ValueError("latent_dim and memory_size must be positive")
        if shrink_threshold is None:
            shrink_threshold = 1.0 / memory_size
        if not 0 <= shrink_threshold <= 1:
            raise ValueError("shrink_threshold must be between 0 and 1")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")

        self.shrink_threshold = shrink_threshold
        self.temperature = float(temperature)
        self.weight = nn.Parameter(torch.empty(memory_size, latent_dim))
        bound = latent_dim ** -0.5
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, z):
        similarity = F.normalize(z, dim=1) @ F.normalize(self.weight, dim=1).T
        attention = F.softmax(similarity / self.temperature, dim=1)

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

    def __init__(self, latent_dim=8, memory_size=100, shrink_threshold=None, temperature=1.0):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28 -> 14
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 14 -> 7
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
        )

        self.memory = MemoryModule(latent_dim, memory_size, shrink_threshold, temperature)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, return_attention=False, return_features=False):
        encoded = self.encoder(x)
        retrieved, attention = self.memory(encoded)
        decoded = self.decoder(retrieved)
        if return_features:
            return decoded, attention, encoded
        if return_attention:
            return decoded, attention
        return decoded


class DINOv2Backbone(nn.Module):
    """Frozen local DINOv2: normalized RGB224 images to L2-normalized CLS384."""

    def __init__(self, source_dir=None, weights_path=None):
        super().__init__()
        assets = Path(__file__).resolve().parent
        source = Path(source_dir or assets / "backbones").resolve()
        weights = Path(weights_path or assets / "backbone_weights" / "dinov2_vits14_pretrain.pth").resolve()
        if not (source / "dinov2" / "hub" / "backbones.py").is_file():
            raise FileNotFoundError(f"Local official DINOv2 source is missing: {source}")
        with weights.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9":
            raise ValueError("DINOv2 weights differ from the fixed pretrained checkpoint")

        os.environ["XFORMERS_DISABLED"] = "1"
        sys.path.insert(0, str(source))
        from dinov2.hub import backbones
        from dinov2.layers.attention import XFORMERS_AVAILABLE

        if Path(backbones.__file__).resolve() != source / "dinov2" / "hub" / "backbones.py":
            raise ValueError("Imported DINOv2 source differs from the requested local source")
        if XFORMERS_AVAILABLE:
            raise ValueError("DINOv2 must use the fixed PyTorch attention backend")
        # Preserve the official img_size=518 default; positions interpolate for RGB224.
        self.backbone = backbones.dinov2_vits14(pretrained=False)
        self.backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (3, 224, 224):
            raise ValueError("DINOv2 expects normalized RGB images shaped [batch, 3, 224, 224]")
        return F.normalize(self.backbone.forward_features(x)["x_norm_clstoken"], dim=1)


class FeatureMemAE(nn.Module):
    """Reconstruct [batch, input_dim] feature vectors through retrieved memory."""

    def __init__(
        self, input_dim=384, hidden_dim=256, latent_dim=64, memory_size=100,
        shrink_threshold=None, temperature=1.0,
    ):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or latent_dim < 1 or memory_size < 1:
            raise ValueError("input_dim, hidden_dim, latent_dim and memory_size must be positive")

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.memory = MemoryModule(latent_dim, memory_size, shrink_threshold, temperature)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, return_attention=False, return_features=False):
        encoded = self.encoder(x)
        retrieved, attention = self.memory(encoded)
        decoded = self.decoder(retrieved)
        if return_features:
            return decoded, attention, encoded
        if return_attention:
            return decoded, attention
        return decoded


class FeatureVAE(nn.Module):
    """Reconstruct frozen features using a diagonal Gaussian latent distribution."""

    def __init__(self, input_dim=384, hidden_dim=256, latent_dim=64):
        super().__init__()
        if min(input_dim, hidden_dim, latent_dim) < 1:
            raise ValueError("input_dim, hidden_dim and latent_dim must be positive")
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, return_distribution=False):
        encoded = self.encoder(x)
        mu, logvar = self.mu(encoded), self.logvar(encoded)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu) if self.training else mu
        decoded = self.decoder(z)
        if return_distribution:
            return decoded, mu, logvar
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
