from pathlib import Path
import hashlib
import json

import torch
from PIL import Image
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

def dinov2_transform():
    """Convert a PIL image to the RGB224 input used by the frozen DINOv2."""
    return transforms.Compose([
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def load_dinov2_features(
    root=ROOT_DIR, device="cuda",
    cache_path=Path(__file__).resolve().parents[1] / "logs/MemAE/_tuning/dinov2_features.pt",
):
    """Validate the existing features, or extract the same fixed split in memory."""
    train_set, normal_set, test_set = load_dataset(root=root, seed=42)
    train, test = train_set.dataset.dataset, test_set.dataset
    train_ids = torch.tensor([train_set.dataset.indices[i] for i in train_set.indices])
    normal_ids = torch.tensor([normal_set.dataset.indices[i] for i in normal_set.indices])
    generator = torch.Generator().manual_seed(1337)
    anomaly_ids = []
    for label in (0, 4, 6):
        available = torch.where(train.targets == label)[0]
        anomaly_ids.append(available[torch.randperm(len(available), generator=generator)[:600]])
    ids = (train_ids, torch.cat([normal_ids] + anomaly_ids), torch.tensor(test_set.indices))
    digest = hashlib.sha256()
    for name, dataset in (("train", train), ("test", test)):
        for field in ("data", "targets"):
            tensor = getattr(dataset, field).contiguous()
            digest.update(f"{name}/{field}/{tensor.dtype}/{tuple(tensor.shape)}".encode())
            digest.update(memoryview(tensor.numpy()).cast("B"))
    expected = {"backbone_name": "dinov2_vits14", "feature_variant": "dinov2_vits14_cls384",
                "feature_dim": 384, "source_data_sha256": digest.hexdigest()}
    cache_path = Path(cache_path)
    if cache_path.exists():
        manifest = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
        if hashlib.sha256(cache_path.read_bytes()).hexdigest() != manifest["cache_sha256"]:
            raise ValueError("DINOv2 feature cache hash differs from its manifest")
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload["metadata"] != {key: value for key, value in manifest.items() if key != "cache_sha256"}:
            raise ValueError("DINOv2 feature cache metadata differs from its manifest")
    else:
        from models import DINOv2Backbone

        backend = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                   torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
                   torch.are_deterministic_algorithms_enabled(),
                   torch.is_deterministic_algorithms_warn_only_enabled())
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
            backbone = DINOv2Backbone().to(device).eval()
            transform = dinov2_transform()
            payload = {"metadata": expected | {
                "backbone_frozen": True, "backbone_mode": "eval", "feature_normalization": "L2 along feature dimension",
                "preprocessing": "PIL RGB; bicubic resize256; crop224; ToTensor; ImageNet normalization",
                "extraction_device": str(device), "extraction_batch_size": 64, "tf32": False,
                "data_split_seed": 42, "tuning_anomaly_seed": 1337,
            }, "normal_val_count": 600}
            with torch.no_grad():
                for name, dataset, indices in zip(("train", "tuning", "test"), (train, train, test), ids):
                    features = []
                    for selected in indices.split(64):
                        images = torch.stack([transform(Image.fromarray(dataset.data[int(i)].numpy())) for i in selected])
                        features.append(backbone(images.to(device)).cpu())
                    payload[name + "_x"] = torch.cat(features)
                    payload[name + "_ids"] = indices
                payload["tuning_labels"] = train.targets[ids[1]].clone()
                payload["test_labels"] = test.targets[ids[2]].clone()
        finally:
            (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = backend[:4]
            torch.use_deterministic_algorithms(backend[4], warn_only=backend[5])
    if any(payload["metadata"].get(key) != value for key, value in expected.items()):
        raise ValueError("DINOv2 cache backbone or source dataset differs")
    if payload["normal_val_count"] != 600:
        raise ValueError("DINOv2 normal validation count differs")
    for name, dataset, indices, count in zip(("train", "tuning", "test"), (train, train, test), ids, (5400, 2400, 4000)):
        cached_ids, features = payload[name + "_ids"], payload[name + "_x"]
        if cached_ids.dtype != torch.long or len(indices) != count or not torch.equal(cached_ids, indices):
            raise ValueError(f"DINOv2 {name} split IDs differ")
        if features.dtype != torch.float32 or features.shape != (count, 384) or not torch.isfinite(features).all():
            raise ValueError(f"Invalid DINOv2 {name} features")
        if not torch.allclose(features.norm(dim=1), torch.ones(count), atol=2e-5, rtol=2e-5):
            raise ValueError(f"DINOv2 {name} features are not L2 normalized")
        if name != "train" and not torch.equal(payload[name + "_labels"], dataset.targets[indices]):
            raise ValueError(f"DINOv2 {name} labels differ")
    return payload


if __name__ == "__main__":
    train_set, val_set, test_set = load_dataset()
    print(
        f"# of train samples: {len(train_set)}, "
        f"# of validation samples: {len(val_set)}, "
        f"# of test samples: {len(test_set)}"
    )
