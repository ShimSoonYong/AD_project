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


if __name__ == "__main__":
    basic_autoencoder = Autoencoder()
    print(basic_autoencoder)
    basic_CNN = CNN()
    print(basic_CNN)
