import torch
from torch import nn
import torch.nn.functional as F


# basic autoencoder
class Autoencoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dim: int = 128,
        activation: str = "relu",
        norm: str = "none",
    ):
        super().__init__()

        def hidden_block(in_dim):
            # Linear -> (LayerNorm) -> activation; defaults keep the original layer layout
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm == "layer":
                layers.append(nn.LayerNorm(hidden_dim))
            # LeakyReLU keeps a small gradient for negative inputs, so units cannot die
            layers.append(nn.LeakyReLU(0.01) if activation == "leaky_relu" else nn.ReLU())
            return layers

        self.encoder = nn.Sequential(
            *hidden_block(28 * 28),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            *hidden_block(latent_dim),
            nn.Linear(hidden_dim, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x.view(-1, 28 * 28))

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        return self.decode(self.encode(x))


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


if __name__ == "__main__":
    basic_autoencoder = Autoencoder()
    print(basic_autoencoder)
    basic_CNN = CNN()
    print(basic_CNN)
