import os
import csv
import json
import random
from pathlib import Path

import numpy as np

print("Running main.py in", os.getcwd())

from argparse import ArgumentParser, Namespace
from data import load_dataset, get_loaders
from torchvision import transforms
from trainer import FeatureTrainer, SSIMLoss, Trainer
from utils import Scorer
from torch import nn, optim
import models
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
        "--threshold",
        type=float,
        default=0.04,
        help="Loss Threshold to classify anomalies",
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
    parser.add_argument("--compactness_weight", type=float, default=0.0,
                        help="Weight of the MemAE feature compactness loss")
    parser.add_argument("--separateness_weight", type=float, default=0.0,
                        help="Weight of the MemAE feature separateness loss")
    parser.add_argument("--memory_margin", type=float, default=0.2,
                        help="Margin between the two nearest normalized memory vectors")
    parser.add_argument("--entropy_weight", type=float, default=1.3e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--memory_size", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--shrink_threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate to pass optimizer"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="Maximum epochs; FeatureMemAE selects the best validation AUROC",
    )
    parser.add_argument(
        "--augment",
        type=list,
        default=None,
        help="A list of torchvision data augmentations to apply.",
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
        help="Root directory for logs; FeatureMemAE uses MemAE/run_n",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers param in PyTorch DataLoader",
    )

    args = parser.parse_args()
    return args


def train_feature_model(args: Namespace):
    """Run the frozen DINOv2 + FeatureMemAE configuration selected in run_126."""
    import matplotlib

    matplotlib.use("Agg")  # Save plots without requiring a desktop GUI backend.
    if args.train_criterion != "MSELoss" or args.score_criterion != "MSELoss":
        raise ValueError("The DINOv2 configuration uses feature MSE for training and scoring")
    if args.compactness_weight or args.separateness_weight or args.augment is not None:
        raise ValueError("The DINOv2 configuration uses no auxiliary losses or augmentation")
    if min(args.epochs, args.batch_size) <= 0 or args.patience < 0:
        raise ValueError("epochs and batch_size must be positive; patience must be non-negative")

    device = torch.device(args.device)
    torch.set_num_threads(2)
    features = data.load_dinov2_features(root=args.data_root, device=device)
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
    model = models.FeatureMemAE(
        input_dim=384, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        memory_size=args.memory_size, shrink_threshold=args.shrink_threshold,
        temperature=args.temperature,
    ).to(device)
    optimizer = getattr(optim, args.optimizer)(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    base = Path(args.log_dir) / "MemAE"
    base.mkdir(parents=True, exist_ok=True)
    number = max((int(p.name[4:]) for p in base.glob("run_*")
                  if p.is_dir() and p.name[4:].isdigit()), default=0) + 1
    while True:
        run = base / f"run_{number:02d}"
        try:
            run.mkdir()
            break
        except FileExistsError:
            number += 1
    print(f"Frozen DINOv2 + FeatureMemAE: {run}", flush=True)
    config = vars(args) | {
        "run": run.name, "model": "FeatureMemAE_DINOv2", "input_dim": 384,
        "feature_dim": 384, "hidden": args.hidden_dim, "latent": args.latent_dim,
        "memory": args.memory_size, "batch": args.batch_size, "entropy": args.entropy_weight,
        "feature_backbone": features["metadata"], "data_split_seed": 42,
        "tuning_anomaly_seed": 1337, "eval_every": 1, "early_stopping_min_epoch": 100,
        "checkpoint_selection": "maximum AUROC on fixed labeled tuning validation",
        "checkpoint_note": "Trainable FeatureMemAE only; DINOv2 weights remain frozen",
        "cudnn_deterministic": True, "cudnn_allow_tf32": True, "matmul_allow_tf32": False,
        "torch_version": str(torch.__version__),
    }
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run / "optimizer_defaults.json").write_text(
        json.dumps(optimizer.defaults, indent=2), encoding="utf-8")
    (run / "split_indices.json").write_text(json.dumps({
        "train_normal_indices": features["train_ids"].tolist(),
        "normal_validation_indices": features["tuning_ids"][:600].tolist(),
        "tuning_validation_train_indices": features["tuning_ids"].tolist(),
        "test_indices": features["test_ids"].tolist(),
    }, indent=2), encoding="utf-8")
    scorer = Scorer(model, [], args.threshold, nn.MSELoss(), str(run), device)
    trainer = FeatureTrainer(
        model, args.epochs, features, optimizer, scorer, device,
        batch_size=args.batch_size, entropy_weight=args.entropy_weight,
        seed=args.seed, patience=args.patience,
    )
    trainer.fit()
    test_metrics, test_scores = trainer.evaluate(
        features["test_x"], features["test_labels"], batch_size=1,
    )
    threshold = float(np.quantile(trainer.tuning_scores[features["tuning_labels"].numpy() == 2], 0.95))
    scorer.THRESHOLDVAL = threshold
    scorer.record = {
        "id": list(range(len(test_scores))),
        "label": (features["test_labels"].numpy() != 2).astype(int).tolist(),
        "score": test_scores.tolist(), "normal": (test_scores <= threshold).astype(int).tolist(),
    }
    scorer.plot_auc(trainer.best_epoch)
    scorer.plot_score(trainer.best_epoch)
    scorer.save_record(trainer.best_epoch)
    for name, scores in (("tuning", trainer.tuning_scores), ("test", test_scores)):
        with (run / f"{name}_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "dataset_index", "class", "anomaly", "score"])
            writer.writerows((i, int(idx), int(label), int(label != 2), float(score))
                             for i, (idx, label, score) in enumerate(zip(
                                 features[name + "_ids"], features[name + "_labels"], scores)))
    config.update(decision_threshold=threshold,
                  threshold_rule="95th percentile of normal validation feature MSE; not used for AUROC")
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics = {
        "status": "complete", "selected_epoch": trainer.best_epoch,
        "completed_epochs": trainer.completed_epochs,
        "early_stopped": trainer.completed_epochs < args.epochs,
        "selected_validation": trainer.selected_validation, "selected_test": test_metrics,
        "final_validation": trainer.final_validation, "test_evaluated": True,
        "selected_checkpoint": "best_MemAE_weight.pth", "final_checkpoint": "final_MemAE_weight.pth",
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"run": run.name, **metrics}, indent=2), flush=True)
    return model


def main(args: Namespace):
    if "FeatureMemAE" in args.models:
        if args.models != ["FeatureMemAE"]:
            raise ValueError("Run FeatureMemAE separately from the image reconstruction models")
        return train_feature_model(args)
    if args.augment == None:
        augmentations = transforms.ToTensor()
    elif isinstance(args.augment, list):
        temp = []
        try:
            for aug in args.augment:
                temp.append(getattr(transforms, aug))
                augmentations = transforms.Compose(temp)
        except AttributeError as e:
            print(e)
            print(f"Please check {args.augment} contains torchvision.transforms")

    try:
        model_list = []
        for m in args.models:
            model_list.append(getattr(models, m))
    except AttributeError as e:
        print(e)
        print(f"Please check {args.models} exist at models.py script.")

    try:
        train_criterion_class = (
            SSIMLoss
            if args.train_criterion == "SSIMLoss"
            else getattr(nn, args.train_criterion)
        )
        score_criterion_class = (
            SSIMLoss
            if args.score_criterion == "SSIMLoss"
            else getattr(nn, args.score_criterion)
        )
    except AttributeError as e:
        print(e)
        print(
            f"Please check {args.train_criterion} and {args.score_criterion} "
            "are PyTorch loss functions or SSIMLoss"
        )
        raise

    try:
        optimizer_class = getattr(optim, args.optimizer)
    except AttributeError as e:
        print(e)
        print(f"Please check {args.optimizer} at PyTorch optimizers")

    device = torch.device(args.device)
    print(f"Running on {args.device}")

    print("Setting directories")
    os.makedirs(args.log_dir, mode=751, exist_ok=True)

    print("Get data loaders")
    train_set, val_set, test_set = load_dataset(root=args.data_root, transform=augmentations)
    train_loader, val_loader, test_loader = get_loaders(
        train_set, val_set, test_set, args.batch_size, args.num_workers
    )

    for model, name in zip(model_list, args.models):
        model = model().to(device=device)
        train_criterion = train_criterion_class().to(device)
        score_criterion = score_criterion_class().to(device)
        optimizer = optimizer_class(model.parameters(), lr=args.lr)

        log_path = args.log_dir + "/"
        os.makedirs(log_path, mode=751, exist_ok=True)
        print("Initializing Scorer")
        scorer = Scorer(
            model=model,
            loader=test_loader,
            threshold=args.threshold,
            criterion=score_criterion,
            log_dir=log_path,
            device=device,
        )

        print("Initializing Trainer")
        trainer = Trainer(
            model=model,
            epochs=args.epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            scorer=scorer,
            device=device,
            compactness_weight=args.compactness_weight,
            separateness_weight=args.separateness_weight,
            memory_margin=args.memory_margin,
        )

        print("Fitting trainer")
        trained_model = trainer.fit()

        print("Evaluating model")
        scorer.collect_record(args.epochs)
        scorer.plot_auc(args.epochs)
        scorer.plot_score(args.epochs)
        scorer.save_record(args.epochs)

        print("Saving model")
        save_path = args.log_dir
        os.makedirs(save_path, mode=751, exist_ok=True)
        torch.save(trained_model.state_dict(), save_path + f"/final_{name}_weight.pth")
        print(f"Final trained model saved at {save_path}")


if __name__ == "__main__":
    print("Start main.py")
    args = get_args()
    main(args)
    print("Finished all procedures")
