from pathlib import Path
import hashlib
import json
from PIL import Image
import torch
from torchvision.transforms import v2
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from utils import *


def load_dataset(
    root: str = ROOT_DIR,
    transform: v2.Transform = v2.ToTensor(),
    seed: int = 42,
):
    trainset = FashionMNIST(root=root, train=True, download=True, transform=transform)
    valset = FashionMNIST(root=root, train=True, download=True, transform=transform)
    testset = FashionMNIST(root=root, train=False, download=True, transform=transform)

    test_label = [0, 2, 4, 6]  # Define anomaly class

    actual_data = torch.isin(trainset.targets, torch.tensor(test_label))
    valset.data = valset.data[actual_data]
    valset.targets = valset.targets[actual_data]

    normal_data = trainset.targets == SELECT_NORMAL
    trainset.data = trainset.data[normal_data]
    trainset.targets = trainset.targets[normal_data]

    g = get_generator(seed)
    train_idx, val_idx = random_split(
        dataset=range(len(trainset)), lengths=[0.9, 0.1], generator=g
    )

    trainset = Subset(trainset, train_idx.indices)
    valset = Subset(valset, val_idx.indices)

    actual_testdata = torch.isin(testset.targets, torch.tensor(test_label))
    testset.data = testset.data[actual_testdata]
    testset.targets = testset.targets[actual_testdata]
    testset = Subset(testset, list(range(len(testset))))

    return trainset, valset, testset


def get_loaders(
    train_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
):
    train_gen = get_generator(seed)
    val_gen = get_generator(seed)
    test_gen = get_generator(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=train_gen,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=val_gen,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=test_gen,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def dinov2_transform():
    """Convert a PIL image to the RGB224 input used by the frozen DINOv2."""
    return v2.Compose(
        [
            v2.Lambda(lambda image: image.convert("RGB")),
            v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def load_dinov2_features(
    root=ROOT_DIR,
    device="cuda",
    transforms=None,
    cache_path=Path(__file__).resolve().parents[1]
    / "logs/MemAE/_tuning/dinov2_features.pt",
    seed=42,
):
    """Validate the existing features, or extract the same fixed split in memory."""
    combined_transform = dinov2_transform()
    if transforms is not None:
        combined_transform = v2.Compose([combined_transform, transforms])

    train_set, val_set, test_set = load_dataset(
        root=root, transform=combined_transform, seed=seed
    )

    train, val, test = train_set.dataset, val_set.dataset, test_set.dataset
    train_ids = torch.tensor(train_set.indices, dtype=torch.long)
    val_ids = torch.tensor(val_set.indices, dtype=torch.long)
    test_ids = torch.tensor(test_set.indices, dtype=torch.long)

    generator = get_generator(seed)
    anomaly_ids = []
    for label in (0, 4, 6):
        available = torch.where(val.targets == label)[0]
        anomaly_ids.append(
            available[torch.randperm(len(available), generator=generator)[:600]]
        )

    tuning_ids = torch.cat([val_ids] + anomaly_ids)
    ids = (train_ids, tuning_ids, test_ids)

    digest = hashlib.sha256()
    for name, dataset in (("train", train), ("test", test)):
        for field in ("data", "targets"):
            tensor = getattr(dataset, field).contiguous()
            digest.update(
                f"{name}/{field}/{tensor.dtype}/{tuple(tensor.shape)}".encode()
            )
            digest.update(memoryview(tensor.numpy()).cast("B"))

    expected = {
        "backbone_name": "dinov2_vits14",
        "feature_variant": "dinov2_vits14_cls384",
        "feature_dim": 384,
        "source_data_sha256": digest.hexdigest(),
    }

    cache_path = Path(cache_path)
    if cache_path.exists():
        manifest = json.loads(
            cache_path.with_suffix(".json").read_text(encoding="utf-8")
        )
        if (
            hashlib.sha256(cache_path.read_bytes()).hexdigest()
            != manifest["cache_sha256"]
        ):
            raise ValueError("DINOv2 feature cache hash differs from its manifest")
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload["metadata"] != {
            key: value for key, value in manifest.items() if key != "cache_sha256"
        }:
            raise ValueError("DINOv2 feature cache metadata differs from its manifest")
    else:
        from models import DINOv2Backbone

        backend = (
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32,
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
            backbone = DINOv2Backbone().to(device).eval()
            payload = {
                "metadata": expected
                | {
                    "backbone_frozen": True,
                    "backbone_mode": "eval",
                    "feature_normalization": "L2 along feature dimension",
                    "preprocessing": "PIL RGB; bicubic resize256; crop224; ToTensor; ImageNet normalization",
                    "extraction_device": str(device),
                    "extraction_batch_size": 64,
                    "tf32": False,
                    "data_split_seed": 42,
                    "tuning_anomaly_seed": 1337,
                },
                "normal_val_count": 600,
            }
            with torch.no_grad():
                for name, dataset, indices in zip(
                    ("train", "tuning", "test"), (train, val, test), ids
                ):
                    features = []
                    for selected in indices.split(64):
                        images = torch.stack(
                            [
                                combined_transform(
                                    Image.fromarray(dataset.data[int(i)].numpy())
                                )
                                for i in selected
                            ]
                        )
                        features.append(backbone(images.to(device)).cpu())
                    payload[name + "_x"] = torch.cat(features)
                    payload[name + "_ids"] = indices
                payload["tuning_labels"] = val.targets[ids[1]].clone()
                payload["test_labels"] = test.targets[ids[2]].clone()
        finally:
            (
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.allow_tf32,
                torch.backends.cuda.matmul.allow_tf32,
            ) = backend[:4]
            torch.use_deterministic_algorithms(backend[4], warn_only=backend[5])

    if any(payload["metadata"].get(key) != value for key, value in expected.items()):
        raise ValueError("DINOv2 cache backbone or source dataset differs")
    if payload["normal_val_count"] != 600:
        raise ValueError("DINOv2 normal validation count differs")
    for name, dataset, indices, count in zip(
        ("train", "tuning", "test"), (train, val, test), ids, (5400, 2400, 4000)
    ):
        cached_ids, features = payload[name + "_ids"], payload[name + "_x"]
        if (
            cached_ids.dtype != torch.long
            or len(indices) != count
            or not torch.equal(cached_ids, indices)
        ):
            raise ValueError(f"DINOv2 {name} split IDs differ")
        if (
            features.dtype != torch.float32
            or features.shape != (count, 384)
            or not torch.isfinite(features).all()
        ):
            raise ValueError(f"Invalid DINOv2 {name} features")
        if not torch.allclose(
            features.norm(dim=1), torch.ones(count), atol=2e-5, rtol=2e-5
        ):
            raise ValueError(f"DINOv2 {name} features are not L2 normalized")
        if name != "train" and not torch.equal(
            payload[name + "_labels"], dataset.targets[indices]
        ):
            raise ValueError(f"DINOv2 {name} labels differ")
    return payload
