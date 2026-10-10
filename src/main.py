import os

os.chdir("/workspace")
print("Running main.py in", os.getcwd())

from argparse import ArgumentParser, Namespace
from data import load_dataset, get_loaders
from torchvision import transforms
from trainer import Trainer
from utils import Scorer
from torch import nn, optim
import models
import torch


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--models",
        type=list,
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
        help="Criterion(Loss) to train model",
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
        "--epochs", type=int, default=100, help="Number of epochs to run"
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
    parser.add_argument("--device", type=str, default="cuda", help="Device type to use")
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
        print(f"Please check {args.model} exist at models.py script.")

    try:
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
    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("Get data loaders")
    train_set, test_set = load_dataset(root=args.data_root, transform=augmentations)
    train_loader, test_loader = get_loaders(
        train_set, test_set, args.batch_size, args.num_workers
    )

    for model, name in zip(model_list, args.models):
        model = model().to(device=device)
        criterion = criterion_class()
        optimizer = optimizer_class(model.parameters(), lr=args.lr)

        print("Initializing Scorer")
        scorer = Scorer(
            model=model,
            loader=test_loader,
            threshold=args.threshold,
            criterion=criterion,
            log_dir=args.log_dir,
            device=device,
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
        )

        print("Fitting trainer")
        trained_model = trainer.fit()

        print("Saving model")
        path = args.log_dir + f"checkpoint/final_{name}.pth"
        torch.save(trained_model.state_dict(), path)
        print(f"Final trained model saved at {path}")


if __name__ == "__main__":
    print("Start main.py")
    args = get_args()
    main(args)
    print("Finished all procedures")
