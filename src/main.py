import os
import csv
import json
import random
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use("Agg")  # Save plots without requiring a desktop GUI backend.

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
        "--mode",
        choices=["train", "inference"],
        default="train",
        help="train or inference",
    )

    parser.add_argument(
        "--threshold_p",
        type=float,
        default=0.95,
        help="Percentile of validation scores to detemine threshold",
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
        default=100,
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

    # Inference Mode
    if args.mode == "inference":
        print(f"Running inference mode for {m}...", flush=True)
        run_dirs = sorted(
            [
                p
                for p in base.glob("run_*")
                if p.is_dir() and (p / "config.json").exists()
            ]
        )
        if not run_dirs:
            raise FileNotFoundError(
                f"No valid trained runs found in {base} for inference."
            )
        latest_run = run_dirs[0]

        config_path = latest_run / "config.json"
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        threshold = config_data.get("decision_threshold", 0.0)

        saved_score_criterion_name = config_data.get(
            "score_criterion", args.score_criterion
        )
        try:
            inference_score_criterion_class = getattr(nn, saved_score_criterion_name)
        except AttributeError:
            try:
                inference_score_criterion_class = getattr(
                    utils, saved_score_criterion_name
                )
            except AttributeError as e:
                raise AttributeError(
                    f"Could not find score_criterion '{saved_score_criterion_name}' in nn or utils."
                ) from e

        inference_score_criterion = inference_score_criterion_class(reduction="none")

        weight_filename = (
            "best_MemAE_weight.pth" if m == "FeatureMemAE" else "best_VAE_weight.pth"
        )
        weight_path = latest_run / weight_filename
        if not weight_path.exists():
            weight_path = latest_run / "final_MemAE_weight.pth"

        print(f"Loading weights from {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

        dummy_optimizer = optimizer_class(model.parameters(), lr=args.lr)

        scorer = Scorer(
            model=model,
            loader=[],
            mode="test",
            threshold=threshold,
            criterion=inference_score_criterion,
            log_dir=str(latest_run),
            device=device,
        )
        scorer.THRESHOLDVAL = threshold

        trainer = FeatureTrainer(
            model=model,
            epochs=args.epochs,
            features=features,
            criterion=inference_score_criterion,
            optimizer=dummy_optimizer,
            scorer=scorer,
            device=device,
            batch_size=args.batch_size,
            entropy_weight=args.entropy_weight,
            seed=args.seed,
            patience=args.patience,
        )

        test_metrics, test_scores = trainer.evaluate(
            features["test_x"],
            features["test_labels"],
            batch_size=1,
        )

        epoch_tag = "inference"
        scorer.record = {
            "id": list(range(len(test_scores))),
            "label": (features["test_labels"].numpy() != 2).astype(int).tolist(),
            "score": test_scores.tolist(),
            "normal": (test_scores <= threshold).astype(int).tolist(),
        }
        scorer.plot_auc(epoch_tag)
        scorer.plot_score(epoch_tag)
        scorer.save_record(epoch_tag)

        with (latest_run / "test_scores_inference.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "dataset_index", "class", "anomaly", "score"])
            writer.writerows(
                (i, int(idx), int(label), int(label != 2), float(score))
                for i, (idx, label, score) in enumerate(
                    zip(features["test_ids"], features["test_labels"], test_scores)
                )
            )

        inference_metrics = {
            "status": "complete_inference",
            "selected_checkpoint": weight_path.name,
            "decision_threshold": threshold,
            "test_metrics": test_metrics,
        }
        (latest_run / "inference_metrics.json").write_text(
            json.dumps(inference_metrics, indent=2), encoding="utf-8"
        )

        print(
            json.dumps(
                {
                    "mode": "inference",
                    "model": m,
                    "run": latest_run.name,
                    "test_metrics": test_metrics,
                },
                indent=2,
            ),
            flush=True,
        )
        return model

    optimizer = optimizer_class(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_criterion = train_criterion_class(reduction="none")
    score_criterion = score_criterion_class(reduction="none")

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
        "model": f"{m}_DINOv2",
        "score_criterion": args.score_criterion,
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
        "checkpoint_note": "Trainable Feature Head only; DINOv2 weights remain frozen",
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
    scorer = Scorer(
        model=model,
        loader=[],
        mode="train",
        threshold=args.threshold_p,
        criterion=score_criterion,
        log_dir=str(run),
        device=device,
    )
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
        np.quantile(
            trainer.tuning_scores[features["tuning_labels"].numpy() == 2],
            args.threshold_p,
        )
    )
    scorer.mode = "test"
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

    for model_cls, name in zip(model_list, args.models):
        model = model_cls().to(device=device)
        train_criterion = train_criterion_class().to(device)
        score_criterion = score_criterion_class().to(device)
        optimizer = optimizer_class(model.parameters(), lr=args.lr)

        log_path = Path(args.log_dir) / name / Path("seed" + str(args.seed))
        os.makedirs(log_path, mode=755, exist_ok=True)

        # Inference
        if args.mode == "inference":
            print(f"Running inference mode for {name}...", flush=True)
            save_path = log_path / "weight"
            weight_file = save_path / f"best_{name}_weight.pth"
            if not weight_file.exists():
                weight_file = save_path / f"final_{name}_weight.pth"
            if not weight_file.exists():
                raise FileNotFoundError(f"Trained weight not found at {weight_file}")

            print(f"Loading weights from {weight_file}")
            model.load_state_dict(torch.load(weight_file, map_location=device))
            model.eval()

            # Threshold loading
            config_file = log_path / "config.json"
            if config_file.exists():
                cfg = json.loads(config_file.read_text(encoding="utf-8"))
                loaded_threshold = cfg.get("decision_threshold", 0.0)
            else:
                loaded_threshold = 0.0

            scorer = Scorer(
                model=model,
                loader=test_loader,
                mode="test",
                threshold=loaded_threshold,
                criterion=score_criterion,
                log_dir=log_path,
                device=device,
            )
            scorer.collect_record()
            scorer.plot_auc("test")
            scorer.plot_score("test")
            scorer.save_record("test")
            print(f"Inference completed for {name}")
            continue

        # Train
        print("Initializing Trainer")
        trainer = Trainer(
            model=model,
            epochs=args.epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            threshold_p=args.threshold_p,
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
            mode="test",
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

        config = vars(args) | {
            "model": name,
            "decision_threshold": float(scorer.THRESHOLDVAL),
            "torch_version": str(torch.__version__),
        }
        (log_path / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    print("Start main.py")
    args = get_args()
    main(args)
    print("Finished all procedures")
