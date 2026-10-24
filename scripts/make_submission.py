"""Build the i-campus code submission notebook from the course submission form.

Fills the "should be filled here" sections of the provided form with self-contained
inference code (copied from src/models.py and src/scoring.py), copies the chosen weight next
to it, and optionally executes the notebook and checks its csv against the training csv.

Example:
    python scripts/make_submission.py --model Autoencoder --latent_dim 4 --hidden_dim 128 \
        --score mse+ssim --weight logs/final_A/Autoencoder/checkpoint/epoch15_weight.pth \
        --check_csv logs/final_A/Autoencoder/record/epoch15_result_team_2.csv
"""

import os
import sys
import json
import uuid
import shutil
import platform
import subprocess
from argparse import ArgumentParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

FILL_MARKER = "should be filled here"
SEPARATOR = "*****************************************************************"


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, help="Model class name in src/models.py")
    parser.add_argument("--weight", required=True, help="Trained weight (.pth) to submit")
    parser.add_argument("--latent_dim", type=int, default=None, help="Same --latent_dim used in training")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Same --hidden_dim used in training")
    parser.add_argument("--activation", default=None, choices=("relu", "leaky_relu"), help="Same --activation used in training")
    parser.add_argument("--norm", default=None, choices=("none", "layer"), help="Same --norm used in training")
    parser.add_argument(
        "--score",
        default="mse",
        choices=("mse", "ssim", "mse+ssim", "perceptual"),
        help="Same --score used in training",
    )
    parser.add_argument("--threshold", type=float, default=0.04, help="Threshold line on histogram")
    parser.add_argument("--team", type=int, default=2, help="Team number")
    parser.add_argument(
        "--members",
        nargs="+",
        default=["이름 (학번)"] * 4,
        help='Team members, e.g. "홍길동 (2020123456)"',
    )
    parser.add_argument(
        "--train_os",
        default=platform.system(),
        help="OS the weight was trained on (FAQ: note it in comments if Linux)",
    )
    parser.add_argument(
        "--train_note",
        default="",
        help="How the weight was trained (loss etc.), written as a comment in the first code cell",
    )
    parser.add_argument(
        "--template",
        default=os.path.join(ROOT, "AD_project_sample", "code submission form.ipynb"),
        help="Path to the provided code submission form",
    )
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "submission"))
    parser.add_argument("--execute", action="store_true", help="Run the notebook after building it")
    parser.add_argument("--check_csv", default=None, help="Training csv the notebook csv must match")
    return parser.parse_args()


def src_source(filename):
    # Copy a src/ file without its __main__ block so the notebook is self-contained
    with open(os.path.join(ROOT, "src", filename), encoding="utf-8") as f:
        source = f.read()
    return source.split('if __name__ == "__main__":')[0].rstrip() + "\n"


def model_kwargs_dict(args):
    kwargs = {}
    if args.latent_dim is not None:
        kwargs["latent_dim"] = args.latent_dim
    if args.hidden_dim is not None:
        kwargs["hidden_dim"] = args.hidden_dim
    if args.activation is not None:
        kwargs["activation"] = args.activation
    if args.norm is not None:
        kwargs["norm"] = args.norm
    return kwargs


def model_kwargs(args):
    return ", ".join(f"{k}={v!r}" for k, v in model_kwargs_dict(args).items())


def code_cells(args, weight_name, csv_name):
    import torch

    train_note = "".join(f"# {line}\n" for line in args.train_note.splitlines() if line.strip())
    inference = [
        f"""# Trained on {args.train_os} / Python {platform.python_version()} / PyTorch {torch.__version__}
{train_note}# This notebook only loads the trained weight and runs inference (no training).
# Device is selected automatically, so it runs on both CPU and GPU.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import torch, torchvision
import torch.nn.functional as F
from torch import nn
from torchvision import transforms, datasets

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Using Device:", DEVICE)""",
        f"""# Settings
TEAM_NUMBER = {args.team}
WEIGHT_PATH = "{weight_name}"  # trained weight placed next to this notebook
SELECT_NORMAL = 2  # label 2 (Pullover) is the normal class
test_label = [0, 2, 4, 6]  # test classes used for evaluation
SCORE = "{args.score}"  # anomaly score: reconstruction error ("mse", "ssim", "mse+ssim" or "perceptual")
THRESHOLDVAL = {args.threshold}  # only used for the histogram line""",
        """# Fashion MNIST test dataset (label 0, 2, 4, 6 only)
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
testset = datasets.FashionMNIST(
    root      = './.data/',
    train     = False,
    download  = True,
    transform = transforms.ToTensor()
)
actual_testdata = torch.isin(testset.targets, torch.tensor(test_label))
testset.data = testset.data[actual_testdata]
testset.targets = testset.targets[actual_testdata]

test_loader = torch.utils.data.DataLoader(
    dataset     = testset,
    batch_size  = 1,
    shuffle     = False,
    num_workers = 0
)
print("# of test samples:", len(testset))

# Normal train data (label 2 only), used only for the mean/std of the "mse+ssim" score
trainset = datasets.FashionMNIST(
    root      = './.data/',
    train     = True,
    download  = True,
    transform = transforms.ToTensor()
)
trainset.data = trainset.data[trainset.targets == SELECT_NORMAL]
trainset.targets = trainset.targets[trainset.targets == SELECT_NORMAL]
stats_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False)""",
        "# Model definitions (copied from src/models.py)\n" + src_source("models.py"),
        "# Reconstruction error functions (copied from src/scoring.py)\n" + src_source("scoring.py"),
        f"""# Load the trained weight
# https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
model = {args.model}({model_kwargs(args)}).to(DEVICE)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(model)

# Mean/std of MSE and SSIM errors on the normal train data (no test data involved)
stats = train_statistics(model, stats_loader, DEVICE) if SCORE == "mse+ssim" else None
print("Train statistics:", stats)""",
        """# Test: anomaly score = reconstruction error of each sample
# Labels are only recorded for the AUROC evaluation, never used to compute the score.
dic_loss = {'id':[], 'label':[], 'score':[], 'normal':[]}

with torch.inference_mode():
    for step, (x, label) in enumerate(test_loader):
        x = x.to(DEVICE)
        decoded = model(x)
        loss = float(anomaly_score(x, decoded, SCORE, stats).cpu().numpy()[0])

        dic_loss['id'].append(step)
        dic_loss['label'].append(int(label != SELECT_NORMAL))
        dic_loss['score'].append(loss)  # anomaly score
        if loss > THRESHOLDVAL: dic_loss['normal'].append('0')
        else: dic_loss['normal'].append('1')""",
        """# Anomaly score histogram
arr_label = np.array(dic_loss['label'])
arr_score = np.array(dic_loss['score'])
score_min = arr_score.min()
score_max = arr_score.max()
plt.hist(arr_score[np.where(arr_label == 0)[0]], bins=30, range=(score_min, score_max), alpha=0.5, label='Normal')
plt.hist(arr_score[np.where(arr_label == 1)[0]], bins=30, range=(score_min, score_max), alpha=0.5, label='Abnormal')
plt.xlabel("Anomaly score")
plt.ylabel("Frequency")
plt.axvline(THRESHOLDVAL, 0, 1, color='red', linestyle='--', linewidth=1)
plt.legend(loc='upper right')
plt.savefig("plot.png")
plt.show()""",
    ]

    graph = [
        """# Generating AUROC
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
fpr, tpr, thresholds = roc_curve(dic_loss['label'], dic_loss['score'], pos_label=1)
auroc = auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.title("AUROC: {}".format(auroc))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.savefig("auroc.png")
plt.show()
print("AUROC: {}".format(auroc))"""
    ]

    dataframe = [
        f"""# Dataframe submitted to kaggle (id, score)
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
df = pd.DataFrame({{'id': dic_loss['id'], 'score': dic_loss['score']}})
df.to_csv("{csv_name}", index=False)  # Index should be not included in the .csv file.
df"""
    ]

    return inference, graph, dataframe


def to_source(text):
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def fill_slots(cells, slots, sources, cell_type):
    # Put sources into the empty form cells of a section, adding cells if the form has too few
    for i, source in enumerate(sources):
        if i < len(slots):
            cell = cells[slots[i]]
        else:
            cell = {"cell_type": cell_type, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": []}
            cells.insert(slots[-1] + 1 + (i - len(slots)), cell)
        cell["cell_type"] = cell_type
        cell["source"] = to_source(source)
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        else:
            cell.pop("execution_count", None)
            cell.pop("outputs", None)


def find_sections(cells):
    # Each section = cells between a "should be filled here" marker and the next separator line
    sections = []
    for i, cell in enumerate(cells):
        if FILL_MARKER in "".join(cell["source"]):
            slots = []
            for j in range(i + 1, len(cells)):
                if "".join(cells[j]["source"]).strip() == SEPARATOR:
                    break
                slots.append(j)
            sections.append(slots)
    return sections


def check_weight(args):
    # Fail early if the weight does not match the model class
    import torch
    import models

    model = getattr(models, args.model)(**model_kwargs_dict(args))
    model.load_state_dict(torch.load(args.weight, map_location="cpu", weights_only=True))


def compare_csv(produced, expected):
    import numpy as np
    import pandas as pd

    a, b = pd.read_csv(produced), pd.read_csv(expected)
    assert list(a.columns) == list(b.columns) == ["id", "score"], (a.columns, b.columns)
    assert len(a) == len(b) and (a["id"] == b["id"]).all(), "id column differs"
    diff = np.abs(a["score"].to_numpy() - b["score"].to_numpy()).max()
    print(f"Max |score difference| vs {expected}: {diff:.3e}")
    if not np.allclose(a["score"], b["score"], rtol=1e-5, atol=1e-7):
        raise SystemExit("Notebook csv does not match the training csv")
    print("Notebook csv matches the training csv")


def main(args):
    args.execute = args.execute or args.check_csv is not None
    check_weight(args)

    with open(args.template, encoding="utf-8") as f:
        nb = json.load(f)

    weight_name = f"model_team{args.team}.pth"
    csv_name = f"result_team_{args.team}.csv"
    team_info = f"> 팀 {args.team}<br>\n" + "<br> ".join(args.members) + "<br>"
    inference, graph, dataframe = code_cells(args, weight_name, csv_name)

    sections = find_sections(nb["cells"])
    assert len(sections) == 4, f"Expected 4 sections in the form, found {len(sections)}"
    # Fill from the last section so inserted cells don't shift earlier indices
    fill_slots(nb["cells"], sections[3], dataframe, "code")
    fill_slots(nb["cells"], sections[2], graph, "code")
    fill_slots(nb["cells"], sections[1], inference, "code")
    fill_slots(nb["cells"], sections[0], [team_info], "markdown")

    os.makedirs(args.out_dir, exist_ok=True)
    nb_path = os.path.join(args.out_dir, f"code_submission_team{args.team}.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    shutil.copyfile(args.weight, os.path.join(args.out_dir, weight_name))
    print(f"Notebook: {nb_path}")
    print(f"Weight:   {os.path.join(args.out_dir, weight_name)}")

    if args.execute:
        # Reuse the already downloaded dataset instead of downloading it again
        local_data = os.path.join(ROOT, "dataset", "FashionMNIST")
        nb_data = os.path.join(args.out_dir, ".data", "FashionMNIST")
        if os.path.isdir(local_data) and not os.path.isdir(nb_data):
            shutil.copytree(local_data, nb_data)
        subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--inplace", nb_path],
            check=True,
        )
        print("Notebook executed without errors")
        if args.check_csv:
            compare_csv(os.path.join(args.out_dir, csv_name), args.check_csv)

    print(f"Upload to i-campus: {os.path.basename(nb_path)}, {weight_name}")


if __name__ == "__main__":
    main(get_args())
