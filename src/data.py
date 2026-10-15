from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, Subset, random_split

ROOT_DIR = Path(__file__).resolve().parents[2] / "02 Fashion MNIST 데이터셋"
SELECT_NORMAL = 2  # Pullover


def load_dataset(root=ROOT_DIR, transform=transforms.ToTensor(), seed=42):
    """Split class 2 training data 9:1; keep test classes 0, 2, 4, 6."""
    train_full = FashionMNIST(
        root=root, train=True, download=False, transform=transform
    )
    test_full = FashionMNIST(
        root=root, train=False, download=False, transform=transform
    )

    train_indices = torch.where(train_full.targets == SELECT_NORMAL)[0].tolist()
    test_classes = torch.tensor([0, 2, 4, 6])
    test_indices = torch.where(torch.isin(test_full.targets, test_classes))[0].tolist()

    normal_set = Subset(train_full, train_indices)
    train_size = int(len(normal_set) * 0.9)
    val_size = len(normal_set) - train_size
    train_set, val_set = random_split(
        normal_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    test_set = Subset(test_full, test_indices)
    return train_set, val_set, test_set


def get_loaders(train_set, val_set, test_set, batch_size, num_workers):

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_set, val_set, test_set = load_dataset()
    print(
        f"# of train samples: {len(train_set)}, "
        f"# of validation samples: {len(val_set)}, "
        f"# of test samples: {len(test_set)}"
    )
