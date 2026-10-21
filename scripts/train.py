"""Train ResNet-18 on a MedMNIST dataset and compare with the paper's table.

28 px: the paper's recipe (100 epochs, lr 1e-3, decay x0.1 at 50 and 75, batch 128, from scratch).
128 px: ImageNet-pretrained ResNet-18, 20 epochs, lr 1e-4 (for the explanations).

usage: uv run scripts/train.py pneumoniamnist --size 28
"""
import argparse
import json
from pathlib import Path

import torch

from whitebox_medmnist.adapters.medmnist_data import INFO, loader
from whitebox_medmnist.application.model import build
from whitebox_medmnist.application.training import fit, predict, scores

# MedMNIST v2 Table (ResNet-18 at 28 px): (ACC, AUC)
PAPER_28 = {"pneumoniamnist": (0.854, 0.944), "bloodmnist": (0.958, 0.998), "pathmnist": (0.907, 0.983)}


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = INFO[args.dataset]
    n_classes, channels = len(info["label"]), info["n_channels"]
    train = loader(args.dataset, "train", args.size, shuffle=True)
    val, test = loader(args.dataset, "val", args.size), loader(args.dataset, "test", args.size)
    model = build(n_classes, channels, args.size).to(device)
    if args.size <= 32:
        fit(model, train, val, epochs=100, lr=1e-3, device=device, milestones=(50, 75))
    else:
        fit(model, train, val, epochs=args.epochs, lr=1e-4, device=device)
    result = {"dataset": args.dataset, "size": args.size, **scores(*predict(model, test, device))}
    if args.size == 28 and args.dataset in PAPER_28:
        result["paper"] = dict(zip(("acc", "auc"), PAPER_28[args.dataset]))
    Path("results/benchmark").mkdir(parents=True, exist_ok=True)
    Path(f"results/benchmark/{args.dataset}_{args.size}.json").write_text(json.dumps(result, indent=1) + "\n")
    Path("data/models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"data/models/{args.dataset}_{args.size}.pt")
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(PAPER_28))
    parser.add_argument("--size", type=int, default=28, choices=[28, 64, 128, 224])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
