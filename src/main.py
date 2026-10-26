import os
import csv
import json
import random
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Save plots without requiring a desktop GUI backend.

import numpy as np

print("Running main.py in", os.getcwd())

from argparse import ArgumentParser, Namespace
from torchvision.transforms import v2
from torch import nn, optim
from trainer import *
from models import *
from utils import *
from data import *
import models
import utils
import torch
import data


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["FeatureMemAE"],
        help="A list of models to train and evaluate",
    )

    parser.add_argument(
        "--train_criterion",
        "--criterion",
        type=str,
        default="MSELoss",
        help="Reconstruction loss for training and validation",
    )
    parser.add_argument(
        "--score_criterion",
        type=str,
        default="MSELoss",
        help="Reconstruction error for evaluation anomaly scores",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Optimizer to Update parameters",
    )
    parser.add_argument(
        "--compactness_weight",
        type=float,
        default=0.0,
        help="Weight of the MemAE feature compactness loss",
    )
    parser.add_argument(
        "--separateness_weight",
        type=float,
        default=0.0,
        help="Weight of the MemAE feature separateness loss",
    )
    parser.add_argument(
        "--memory_margin",
        type=float,
        default=0.2,
        help="Margin between the two nearest normalized memory vectors",
    )
    parser.add_argument("--entropy_weight", type=float, default=1.3e-6)
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=1e-5,
        help="FeatureVAE KL weight; KL sums latent dimensions and averages samples",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--memory_size", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--shrink_threshold", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate to pass optimizer"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum epochs; feature models select the best validation AUROC",
    )
    parser.add_argument(
        "--augment",
        nargs="+",
        default=["none"],
        help="A list of torchvision data augmentations to apply. Default is 'none'",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size on training. Default is 64",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device type to use")
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(data.ROOT_DIR),
        help="Root directory to store FashionMNIST dataset. Default is 'dataset'",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "logs"),
        help="Root directory for logs; FeatureVAE uses DINOVAE/run_n, FeatureMemAE uses MemAE/run_n",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers param in PyTorch DataLoader",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="PyTorch random seed for reproducibility"
    )

    args = parser.parse_args()
    return args


def run_fae(args: Namespace, m: str):

    if args.augment == ["none"]:
        augmentations = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])
    elif isinstance(args.augment, list):
        temp = []
        try:
            for aug in args.augment:
                temp.append(getattr(v2, aug))
                augmentations = v2.Compose(temp)
        except AttributeError as e:
            print(e)

    try:
        train_criterion_class = getattr(nn, args.train_criterion)
    except AttributeError as e:
        print(e)
        try:
            train_criterion_class = getattr(utils, args.train_criterion)
        except AttributeError as e:
            print(e)
            print(f"Please check {args.train_criterion} exist at `utils.py` script.")

    try:
        score_criterion_class = getattr(nn, args.score_criterion)
    except AttributeError as e:
        print(e)
        try:
            score_criterion_class = getattr(utils, args.score_criterion)
        except AttributeError as e:
            print(e)
            print(f"Please check {args.score_criterion} exist at `utils.py` script.")

    try:
        optimizer_class = getattr(optim, args.optimizer)
    except AttributeError as e:
        print(e)

    device = torch.device(args.device)
    torch.set_num_threads(2)
    features = data.load_dinov2_features(
        root=args.data_root, transforms=augmentations, device=device
    )
    # Reset after backbone initialization/extraction so the trainable head matches run_126.

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False

    if m == "FeatureVAE":
        model = models.FeatureVAE(
            input_dim=384,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
        ).to(device)
        base = Path(args.log_dir) / "DINOVAE" / Path("seed" + str(args.seed))
        base.mkdir(parents=True, exist_ok=True)
    elif m == "FeatureMemAE":
        model = models.FeatureMemAE(
            input_dim=384,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            memory_size=args.memory_size,
            shrink_threshold=args.shrink_threshold,
            temperature=args.temperature,
        ).to(device)
        base = Path(args.log_dir) / "MemAE" / Path("seed" + str(args.seed))
        base.mkdir(parents=True, exist_ok=True)
    else:
        raise NotImplementedError(f"{m} is not implemented in `main.py`")

    optimizer = optimizer_class(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_criterion = train_criterion_class(reduction="none")

    number = (
        max(
            (
                int(p.name[4:])
                for p in base.glob("run_*")
                if p.is_dir() and p.name[4:].isdigit()
            ),
            default=0,
        )
        + 1
    )
    while True:
        run = base / f"run_{number:02d}"
        try:
            run.mkdir()
            break
        except FileExistsError:
            number += 1

    print(f"Frozen DINOv2 + FeatureMemAE: {run}", flush=True)

    config = vars(args) | {
        "run": run.name,
        "model": "FeatureMemAE_DINOv2",
        "input_dim": 384,
        "feature_dim": 384,
        "hidden": args.hidden_dim,
        "latent": args.latent_dim,
        "memory": args.memory_size,
        "batch": args.batch_size,
        "entropy": args.entropy_weight,
        "feature_backbone": features["metadata"],
        "data_split_seed": 42,
        "tuning_anomaly_seed": 1337,
        "eval_every": 1,
        "early_stopping_min_epoch": 100,
        "checkpoint_selection": "maximum AUROC on fixed labeled tuning validation",
        "checkpoint_note": "Trainable FeatureMemAE only; DINOv2 weights remain frozen",
        "cudnn_deterministic": True,
        "cudnn_allow_tf32": True,
        "matmul_allow_tf32": False,
        "torch_version": str(torch.__version__),
    }
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run / "optimizer_defaults.json").write_text(
        json.dumps(optimizer.defaults, indent=2), encoding="utf-8"
    )
    (run / "split_indices.json").write_text(
        json.dumps(
            {
                "train_normal_indices": features["train_ids"].tolist(),
                "normal_validation_indices": features["tuning_ids"][:600].tolist(),
                "tuning_validation_train_indices": features["tuning_ids"].tolist(),
                "test_indices": features["test_ids"].tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    score_criterion = score_criterion_class(reduction="none")
    scorer = Scorer(model, [], score_criterion, str(run), device)
    trainer = FeatureTrainer(
        model,
        args.epochs,
        features,
        train_criterion,
        optimizer,
        scorer,
        device,
        batch_size=args.batch_size,
        entropy_weight=args.entropy_weight,
        seed=args.seed,
        patience=args.patience,
    )
    trainer.fit()
    test_metrics, test_scores = trainer.evaluate(
        features["test_x"],
        features["test_labels"],
        batch_size=1,
    )
    threshold = float(
        np.quantile(trainer.tuning_scores[features["tuning_labels"].numpy() == 2], 0.95)
    )
    scorer.THRESHOLDVAL = threshold
    scorer.record = {
        "id": list(range(len(test_scores))),
        "label": (features["test_labels"].numpy() != 2).astype(int).tolist(),
        "score": test_scores.tolist(),
        "normal": (test_scores <= threshold).astype(int).tolist(),
    }
    scorer.plot_auc(trainer.best_epoch)
    scorer.plot_score(trainer.best_epoch)
    scorer.save_record(trainer.best_epoch)
    for name, scores in (("tuning", trainer.tuning_scores), ("test", test_scores)):
        with (run / f"{name}_scores.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "dataset_index", "class", "anomaly", "score"])
            writer.writerows(
                (i, int(idx), int(label), int(label != 2), float(score))
                for i, (idx, label, score) in enumerate(
                    zip(features[name + "_ids"], features[name + "_labels"], scores)
                )
            )
    config.update(
        decision_threshold=threshold,
        threshold_rule="95th percentile of normal validation feature MSE; not used for AUROC",
    )
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics = {
        "status": "complete",
        "selected_epoch": trainer.best_epoch,
        "completed_epochs": trainer.completed_epochs,
        "early_stopped": trainer.completed_epochs < args.epochs,
        "selected_validation": trainer.selected_validation,
        "selected_test": test_metrics,
        "final_validation": trainer.final_validation,
        "test_evaluated": True,
        "selected_checkpoint": "best_MemAE_weight.pth",
        "final_checkpoint": "final_MemAE_weight.pth",
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"run": run.name, **metrics}, indent=2), flush=True)
    return model


def main(args: Namespace):
    if min(args.epochs, args.batch_size) <= 0 or args.patience < 0:
        raise ValueError(
            "epochs and batch_size must be positive; patience must be non-negative"
        )

    if args.augment == ["none"]:
        augmentations = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])
    elif isinstance(args.augment, list):
        temp = []
        try:
            for aug in args.augment:
                temp.append(getattr(v2, aug))
                augmentations = v2.Compose(temp)
        except AttributeError as e:
            print(e)

    try:
        train_criterion_class = getattr(nn, args.train_criterion)
    except AttributeError as e:
        print(e)
        try:
            train_criterion_class = getattr(utils, args.train_criterion)
        except AttributeError as e:
            print(e)
            print(f"Please check {args.train_criterion} exist at `utils.py` script.")

    try:
        score_criterion_class = getattr(nn, args.score_criterion)
    except AttributeError as e:
        print(e)
        try:
            score_criterion_class = getattr(utils, args.score_criterion)
        except AttributeError as e:
            print(e)
            print(f"Please check {args.score_criterion} exist at `utils.py` script.")

    try:
        optimizer_class = getattr(optim, args.optimizer)
    except AttributeError as e:
        print(e)

    try:
        model_list = []
        for m in args.models:
            if m in ("FeatureMemAE", "FeatureVAE"):
                run_fae(args, m)
                continue

            model_list.append(getattr(models, m))
    except AttributeError as e:
        print(e)
        print(f"Please check {args.models} exist at `models` dir")

    device = torch.device(args.device)
    print(f"Running on {args.device}")

    print("Setting directories")
    os.makedirs(args.data_root, mode=755, exist_ok=True)
    os.makedirs(args.log_dir, mode=755, exist_ok=True)

    print("Get data loaders")
    train_set, val_set, test_set = load_dataset(
        root=args.data_root, transform=augmentations, seed=args.seed
    )
    train_loader, val_loader, test_loader = get_loaders(
        train_set, val_set, test_set, args.batch_size, args.num_workers
    )

    for model, name in zip(model_list, args.models):
        model = model().to(device=device)
        train_criterion = train_criterion_class().to(device)
        score_criterion = score_criterion_class().to(device)
        optimizer = optimizer_class(model.parameters(), lr=args.lr)

        log_path = Path(args.log_dir) / name / Path("seed" + str(args.seed))
        os.makedirs(log_path, mode=755, exist_ok=True)

        print("Initializing Trainer")
        trainer = Trainer(
            model=model,
            epochs=args.epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            log_dir=log_path,
            device=device,
            compactness_weight=args.compactness_weight,
            separateness_weight=args.separateness_weight,
            memory_margin=args.memory_margin,
        )

        print("Fitting trainer")
        trained_model = trainer.fit()

        print("Testing model")
        print("Initializing Scorer")
        scorer = Scorer(
            model=trained_model,
            loader=test_loader,
            threshold=trainer.scorer.THRESHOLDVAL,  # Use last val threshold
            criterion=score_criterion,
            log_dir=log_path,
            device=device,
        )
        scorer.collect_record()
        scorer.plot_auc("test")
        scorer.plot_score("test")
        scorer.save_record("test")

        print("Saving model")
        save_path = log_path / "weight"
        os.makedirs(save_path, mode=755, exist_ok=True)
        torch.save(trained_model.state_dict(), save_path / f"final_{name}_weight.pth")
        print(f"Final trained model saved at {save_path}")


if __name__ == "__main__":
    print("Start main.py")
    args = get_args()
    main(args)
    print("Finished all procedures")
