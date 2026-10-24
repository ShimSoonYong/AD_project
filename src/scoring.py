import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

SCORES = ("mse", "ssim", "mse+ssim", "perceptual")


def gaussian_window(size: int = 7, sigma: float = 1.5):
    g = torch.exp(-((torch.arange(size) - size // 2) ** 2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def ssim_map(a: torch.Tensor, b: torch.Tensor, C1: float = 0.01**2, C2: float = 0.03**2):
    # SSIM (Wang et al., 2004) with a 7x7 Gaussian window; a, b: (N, 1, 28, 28) in [0, 1]
    # Always computed in float32: the variance terms are too imprecise under fp16 autocast
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, b = a.float(), b.float()
        window = gaussian_window().to(a.device)
        mu_a, mu_b = F.conv2d(a, window), F.conv2d(b, window)
        var_a = F.conv2d(a * a, window) - mu_a**2
        var_b = F.conv2d(b * b, window) - mu_b**2
        cov = F.conv2d(a * b, window) - mu_a * mu_b
        return ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / (
            (mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2)
        )


class SSIMLoss(nn.Module):
    # Training loss: 1 - mean SSIM between input and reconstruction
    def forward(self, decoded: torch.Tensor, x: torch.Tensor):
        return (1 - ssim_map(x, decoded)).mean()


def base_scores(x: torch.Tensor, decoded: torch.Tensor):
    # Per-sample reconstruction errors, computed the same way for every sample
    return {
        "mse": ((decoded - x) ** 2).flatten(1).mean(1),
        "ssim": (1 - ssim_map(x, decoded)).flatten(1).mean(1),
    }


@torch.inference_mode()
def train_statistics(model: nn.Module, loader, device: torch.device):
    # Mean/std of each error on the normal TRAIN data, used to put MSE and SSIM on one scale
    collected = {"mse": [], "ssim": []}
    for x, _ in loader:
        x = x.to(device)
        for name, value in base_scores(x, model(x)).items():
            collected[name].append(value)
    return {
        name: (torch.cat(values).mean().item(), torch.cat(values).std().item())
        for name, values in collected.items()
    }


_feature_net = {}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def resnet_features(x: torch.Tensor):
    # ImageNet-pretrained ResNet18 (frozen, not trained here), layer1-3 features of a 112x112 copy
    if x.device not in _feature_net:
        # fork_rng: building the network must not consume the training random state
        with torch.random.fork_rng(devices=[]):
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.requires_grad_(False)  # frozen: only used to measure distances
        _feature_net[x.device] = net.to(x.device).eval()
    net = _feature_net[x.device]
    x = F.interpolate(x.float(), size=112, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    features = []
    for layer in (net.layer1, net.layer2, net.layer3):
        x = layer(x)
        features.append(F.normalize(x, dim=1))  # unit length over channels at each position
    return features


def perceptual_distance(x: torch.Tensor, decoded: torch.Tensor):
    # Reconstruction error in a pretrained feature space (LPIPS-style): input vs its reconstruction.
    # Used as the anomaly score, and (with gradients through `decoded`) as an extra training loss
    with torch.no_grad():
        target = resnet_features(x)
    return sum(
        ((a - b) ** 2).sum(1).mean((1, 2))
        for a, b in zip(target, resnet_features(decoded))
    )


def anomaly_score(x: torch.Tensor, decoded: torch.Tensor, kind: str, stats=None):
    if kind == "perceptual":
        return perceptual_distance(x, decoded)
    scores = base_scores(x, decoded)
    if kind == "mse+ssim":
        return sum((scores[k] - stats[k][0]) / stats[k][1] for k in ("mse", "ssim"))
    return scores[kind]
