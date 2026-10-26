import torch
from torch import nn
from math import sqrt
from .component import *
from einops import rearrange
import torch.nn.functional as F
from einops.layers.torch import Rearrange


# basic autoencoder
class AE(nn.Module):
    def __init__(self, features: int = 28 * 28):
        super().__init__()

        self.encoder = nn.Sequential(
            Rearrange("n c h w -> n (c h w)"),
            nn.Linear(features, features // 4, bias=False),
            nn.ReLU(),
            nn.Linear(features // 4, features // 16, bias=False),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(features // 16, features // 4, bias=False),
            nn.ReLU(),
            nn.Linear(features // 4, features, bias=False),
            Rearrange(
                "n (c h w) -> n c h w",
                c=1,
                h=int(sqrt(features)),
                w=int(sqrt(features)),
            ),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


# basic sparse autoencoder
class SAE(nn.Module):
    def __init__(self, features: int = 28 * 28):
        super().__init__()

        self.encoder = nn.Sequential(
            Rearrange("n c h w -> n (c h w)"),
            nn.Linear(features, 4 * features, bias=False),
            nn.ReLU(),
            nn.Linear(4 * features, 32 * features, bias=False),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(32 * features, 4 * features, bias=False),
            nn.ReLU(),
            nn.Linear(4 * features, features, bias=False),
            Rearrange(
                "n (c h w) -> n c h w",
                c=1,
                h=int(sqrt(features)),
                w=int(sqrt(features)),
            ),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


# Sparse AE with attention
class SAttn(nn.Module):
    def __init__(self, channels: int = 1):
        super().__init__()

        self.encoder = nn.Sequential(
            AxialAttention(channels, 4 * channels),
            Rearrange("b c h w -> b h w c"),
            nn.LayerNorm(4 * channels),
            Rearrange("b h w c -> b c h w"),
            AxialAttention(4 * channels, 16 * channels),
        )

        self.decoder = nn.Sequential(
            AxialAttention(16 * channels, 4 * channels),
            Rearrange("b c h w -> b h w c"),
            nn.LayerNorm(4 * channels),
            Rearrange("b h w c -> b c h w"),
            AxialAttention(4 * channels, channels),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


# basic Convolutional AE
class CAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU())
        self.pool = nn.MaxPool2d(2, 2)

        self.embed = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.ConvTranspose2d(4, 8, 3, 1, 1, 0),
            nn.ConvTranspose2d(8, 16, 3, 1, 1, 0),
            nn.ReLU(),
        )

        self.unpool = nn.Upsample(scale_factor=2, mode="bilinear")
        self.decoder = nn.ConvTranspose2d(16, 1, 3, 1, 1, 0)

    def forward(self, x):
        encoded = self.encoder(x)
        pooled = self.pool(encoded).contiguous()
        embed = self.embed(pooled).contiguous()
        unpooled = self.unpool(embed + pooled)  # Simple skip-connection
        decoded = self.decoder(unpooled)

        return decoded


class PCR(nn.Module):
    def __init__(self, features: int = 28, k: int = 8):
        super().__init__()

        self.features = features
        self.pca = ImagePCA(k)
        self.reg = nn.Sequential(
            nn.Linear(k * features, features * features),
            Rearrange("b (c h w) -> b c h w", c=1, h=features, w=features),
        )

    def forward(self, x):
        encoded = self.pca(x)
        decoded = self.reg(encoded)

        return decoded


class PCAE(nn.Module):
    def __init__(self, features: int = 28 * 28, k: int = 8):
        super().__init__()

        self.pcr = PCR(features=int(sqrt(features)), k=k)
        self.ae = AE(features)

    def forward(self, x):
        encoded = self.pcr(x)
        decoded = self.ae(encoded)

        return decoded


class PCSAE(nn.Module):
    def __init__(self, features: int = 28 * 28, k: int = 8):
        super().__init__()

        self.pcr = PCR(features=int(sqrt(features)), k=k)
        self.sae = SAE(features)

    def forward(self, x):
        encoded = self.pcr(x)
        decoded = self.sae(encoded)

        return decoded


class PCCAE(nn.Module):
    def __init__(self, features: int = 28 * 28, k: int = 8):
        super().__init__()

        self.pcr = PCR(features=int(sqrt(features)), k=k)
        self.cae = CAE()

    def forward(self, x):
        encoded = self.pcr(x)
        decoded = self.cae(encoded)

        return decoded


class PCAttn(nn.Module):
    def __init__(self, features: int = 28 * 28, k: int = 8):
        super().__init__()

        self.pcr = PCR(features=int(sqrt(features)), k=k)
        self.attn = SAttn()

    def forward(self, x):
        encoded = self.pcr(x)
        decoded = self.attn(encoded)

        return decoded


class ConvSumAE(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 2 * in_channels, 3, 1, 1), nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2)

        self.embed = nn.Sequential(
            ConvSum2d(2 * in_channels, 4 * in_channels, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            ConvTransposeSum2d(4 * in_channels, 2 * out_channels, 3, 1, 1, 0),
            nn.ReLU(),
        )

        self.unpool = nn.Upsample(scale_factor=2, mode="bilinear")
        self.decoder = ConvTransposeSum2d(2 * out_channels, out_channels, 3, 1, 1, 0)

    def forward(self, x):
        encoded = self.encoder(x)
        pooled = self.pool(encoded).contiguous()
        embed = self.embed(pooled).contiguous()
        unpooled = self.unpool(embed + pooled)  # Simple skip-connection
        decoded = self.decoder(unpooled)

        return decoded


class PCConvSumAE(nn.Module):
    def __init__(self, features: int = 28 * 28, k: int = 8):
        super().__init__()

        self.pcr = PCR(features=int(sqrt(features)), k=k)
        self.ae = ConvSumAE()

    def forward(self, x):
        encoded = self.pcr(x)
        decoded = self.ae(encoded)

        return decoded


if __name__ == "__main__":
    ae = AE().cuda()
    print(ae)
    sae = SAE().cuda()
    print(sae)
    sattn = SAttn().cuda()
    print(sattn)
    cae = CAE().cuda()
    print(cae)
    pcr = PCR().cuda()
    print(pcr)
    pcae = PCAE().cuda()
    print(pcae)
    pcsae = PCSAE().cuda()
    print(pcsae)
    pccae = PCCAE().cuda()
    print(pccae)
    pcattn = PCAttn().cuda()
    print(pcattn)
    convsumae = ConvSumAE()
    print(convsumae)
    pcconvsumae = PCConvSumAE()
    print(pcconvsumae)
