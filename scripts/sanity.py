"""Sanity checks (Adebayo et al., 2018) and per-image stability for the six saliency methods on a trained model.

  cascading:    re-initialize fc, then layer4, ... down to conv1, and compare each map with the trained model's
  independent:  re-initialize one block at a time
  stability:    the trained model on 5 slightly noisy copies of each image: mean top-5% overlap and Spearman

usage: uv run scripts/sanity.py pneumoniamnist --size 128 --n 50
"""
import argparse
import copy
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from whitebox_medmnist.adapters.medmnist_data import INFO, arrays
from whitebox_medmnist.application.model import BLOCKS, build, randomize
from whitebox_medmnist.application.saliency import METHODS, explain
from whitebox_medmnist.application.similarity import all_measures, spearman_abs
from whitebox_medmnist.domain.trust import passes_sanity, top_k_overlap


def maps_for(model, x, target, device, chunk=5):
    """In small chunks: Integrated Gradients alone runs 32 forward and backward passes per image."""
    out = {}
    for m in METHODS:
        parts = [explain(model, x[i:i + chunk].to(device), target[i:i + chunk].to(device), m).cpu().numpy()
                 for i in range(0, len(x), chunk)]
        out[m] = np.concatenate(parts)
    return out


def compare(reference, other):
    """Mean of each similarity measure over images, per method.

    Grad-CAM keeps only positive evidence, so some maps come out all zeros; a rank correlation with a constant map is
    undefined. Those images are left out of the means and counted in `constant_share`.
    """
    out = {}
    for m in METHODS:
        per_image = [all_measures(a, b) for a, b in zip(reference[m], other[m])]
        constant = np.mean([np.ptp(a) == 0 or np.ptp(b) == 0 for a, b in zip(reference[m], other[m])])
        out[m] = {k: float(np.nanmean([p[k] for p in per_image])) for k in per_image[0]}
        out[m]["constant_share"] = float(constant)
    return out


def main(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    info = INFO[args.dataset]
    x, y = arrays(args.dataset, "test", args.size)
    idx = np.random.default_rng(args.seed).choice(len(x), args.n, replace=False)
    x, y = torch.from_numpy(x[idx]), torch.from_numpy(y[idx])
    trained = build(len(info["label"]), info["n_channels"], args.size).to(device)
    trained.load_state_dict(torch.load(f"data/models/{args.dataset}_{args.size}.pt", map_location=device))
    with torch.no_grad():
        target = trained(x.to(device)).argmax(1).cpu()  # explain what the trained model predicted
    reference = maps_for(trained, x, target, device)

    rows = []
    for depth in range(1, len(BLOCKS) + 1):
        model = randomize(copy.deepcopy(trained), BLOCKS[:depth], seed=args.seed)
        for m, measures in compare(reference, maps_for(model, x, target, device)).items():
            rows.append({"test": "cascading", "randomized": "+".join(BLOCKS[:depth]), "method": m, **measures})
    for block in BLOCKS:
        model = randomize(copy.deepcopy(trained), [block], seed=args.seed)
        for m, measures in compare(reference, maps_for(model, x, target, device)).items():
            rows.append({"test": "independent", "randomized": block, "method": m, **measures})

    stability = {}
    gen = torch.Generator().manual_seed(args.seed)
    k = int(0.05 * args.size * args.size)
    noisy = [maps_for(trained, (x + args.noise * torch.randn(x.shape, generator=gen)).clamp(0, 1), target, device)
             for _ in range(5)]
    for m in METHODS:
        overlaps, ranks = [], []
        for i in range(args.n):
            for a, b in itertools.combinations([n[m][i] for n in noisy], 2):
                overlaps.append(top_k_overlap(a.ravel().tolist(), b.ravel().tolist(), k))
                ranks.append(spearman_abs(a, b))
        stability[m] = {"top5pct_overlap": float(np.mean(overlaps)), "spearman_abs": float(np.nanmean(ranks))}

    top = {r["method"]: r["spearman_abs"] for r in rows if r["test"] == "cascading" and r["randomized"] == "fc+layer4"}
    verdict = {m: {"similarity_after_top_randomized": top[m], "passes_sanity": passes_sanity(top[m]),
                   **stability[m]} for m in METHODS}
    out = Path("results/sanity")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.dataset}_{args.size}_randomization.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / f"{args.dataset}_{args.size}_verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
    for m, v in verdict.items():
        print(f"  {m:22} after fc+layer4 randomized: {v['similarity_after_top_randomized']:.2f}  "
              f"{'PASS' if v['passes_sanity'] else 'FAIL'}   stability top-5%: {v['top5pct_overlap']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.02, help="std of the noise for the stability test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu to leave the GPU to another job")
    main(parser.parse_args())
