import os
import random
import numpy as np
from argparse import ArgumentParser, Namespace
from data import load_dataset, get_loaders
from torchvision import transforms
from trainer import Trainer
from utils import Scorer
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import models
import scoring
import torch


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Autoencoder"],
        help="A list of models to train and evaluate",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.04,
        help="Loss Threshold to classify anomalies",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="MSELoss",
        help="Criterion(Loss) to train model: a torch.nn loss or SSIMLoss (scoring.py)",
    )
    parser.add_argument(
        "--score",
        type=str,
        default="mse",
        choices=scoring.SCORES,
        help="Anomaly score of each test sample. Default is mse",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Optimizer to Update parameters",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate to pass optimizer"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to run, default is 100",
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
        default=32,
        help="Batch size on training. Default is 32",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device type to use. Default is cuda if available, otherwise cpu",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="dataset",
        help="Root directory to store FashionMNIST dataset. Default is 'dataset'",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Root directoru to store log files from Scorer",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="num_workers param in PyTorch DataLoader",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Latent dimension passed to models that accept it (e.g. Autoencoder)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Hidden layer size passed to models that accept it (e.g. Autoencoder)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        choices=("relu", "leaky_relu"),
        help="Hidden activation of models that accept it (e.g. Autoencoder). Default is relu",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default=None,
        choices=("none", "layer"),
        help="'layer' adds LayerNorm before each hidden activation (e.g. Autoencoder). Default is none",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.0,
        help="Training only: probability of zeroing each 7x7 input block. Default is 0 (no masking)",
    )
    parser.add_argument(
        "--perceptual_weight",
        type=float,
        default=0.0,
        help="Weight of the extra perceptual (ImageNet ResNet18 feature) training loss. Default is 0",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.0,
        help="Fraction of the normal train data held out as validation for early stopping. Default is 0",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Stop when the validation loss has not improved for this many epochs (needs --val_ratio). "
        "Default is 0 (no early stopping)",
    )
    parser.add_argument(
        "--center_weight",
        type=float,
        default=0.0,
        help="Weight of the latent center loss ||z - c||^2 (c = mean train latent, per epoch). "
        "Needs a model with encode/decode (Autoencoder). Default is 0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible runs. Default is no fixed seed",
    )

    args = parser.parse_args()
    return args


def main(args: Namespace):
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
        if hasattr(scoring, args.criterion):
            criterion_class = getattr(scoring, args.criterion)
        else:
            criterion_class = getattr(nn, args.criterion)
    except AttributeError as e:
        print(e)
        print(f"Please check {args.criterion} exist at PyTorch loss functions")

    try:
        optimizer_class = getattr(optim, args.optimizer)
    except AttributeError as e:
        print(e)
        print(f"Please check {args.optimizer} at PyTorch optimizers")

    device = torch.device(args.device)
    print(f"Running on {args.device}")

    print("Setting directories")
    os.makedirs(args.data_root, mode=751, exist_ok=True)
    os.makedirs(args.log_dir, mode=751, exist_ok=True)

    print("Get data loaders")
    train_set, test_set = load_dataset(root=args.data_root, transform=augmentations)
    val_loader = None
    if args.val_ratio > 0:
        # Hold out part of the NORMAL train data for early stopping (test data is never used)
        n_val = int(len(train_set) * args.val_ratio)
        split_generator = torch.Generator().manual_seed(args.seed if args.seed is not None else 0)
        train_set, val_set = random_split(
            train_set, [len(train_set) - n_val, n_val], generator=split_generator
        )
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
        print(f"Train / validation split of normal train data: {len(train_set)} / {n_val}")
    train_loader, test_loader = get_loaders(
        train_set, test_set, args.batch_size, args.num_workers
    )

    # Normal train data in a fixed order, for the train statistics of the "mse+ssim" score
    stats_loader = DataLoader(train_set, batch_size=256, shuffle=False)

    model_kwargs = {}
    if args.latent_dim is not None:
        model_kwargs["latent_dim"] = args.latent_dim
    if args.hidden_dim is not None:
        model_kwargs["hidden_dim"] = args.hidden_dim
    if args.activation is not None:
        model_kwargs["activation"] = args.activation
    if args.norm is not None:
        model_kwargs["norm"] = args.norm

    for model, name in zip(model_list, args.models):
        if args.seed is not None:
            # Same initial weights and data order for every model in this run
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
        model = model(**model_kwargs).to(device=device)
        criterion = criterion_class()
        optimizer = optimizer_class(model.parameters(), lr=args.lr)

        log_path = args.log_dir + "/" + name
        save_path = log_path + "/checkpoint"
        os.makedirs(save_path, mode=751, exist_ok=True)
        print("Initializing Scorer")
        scorer = Scorer(
            model=model,
            loader=test_loader,
            threshold=args.threshold,
            log_dir=log_path,
            device=device,
            score=args.score,
            stats_loader=stats_loader,
        )

        print("Initializing Trainer")
        trainer = Trainer(
            model=model,
            epochs=args.epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scorer=scorer,
            device=device,
            checkpoint_dir=save_path,
            mask_ratio=args.mask_ratio,
            perceptual_weight=args.perceptual_weight,
            center_weight=args.center_weight,
            val_loader=val_loader,
            patience=args.patience,
        )

        print("Fitting trainer")
        trained_model = trainer.fit()

        print("Saving model")
        torch.save(trained_model.state_dict(), save_path + f"/final_{name}_weight.pth")
        print(f"Final trained model saved at {save_path}")


if __name__ == "__main__":
    # Move to the project root (/workspace in Docker) so relative paths work anywhere.
    # Kept under __main__ so spawned DataLoader workers (Windows) don't repeat it.
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Running main.py in", os.getcwd())
    print("Start main.py")
    args = get_args()
    main(args)
    print("Finished all procedures")
