"""Would we notice if the images changed? Evidently compares per-image signals (brightness, contrast, sharpness,
entropy, model confidence) of new images against the validation set, for clean images and three corruptions,
and we check whether the alarm goes off when the model's accuracy actually drops.

usage: uv run --group mlops scripts/drift.py pneumoniamnist --size 128
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evidently import Report
from evidently.presets import DataDriftPreset

from whitebox_medmnist.adapters.medmnist_data import INFO, arrays
from whitebox_medmnist.application.drift import corrupt, features
from whitebox_medmnist.application.model import build


@torch.no_grad()
def predict(model, x, device):
    probs = torch.cat([model(torch.from_numpy(x[i:i + 256]).to(device)).softmax(-1).cpu()
                       for i in range(0, len(x), 256)])
    return probs.numpy()


def frame(model, x, device):
    probs = predict(model, x, device)
    return pd.DataFrame(features(x) | {"confidence": probs.max(1)}), probs.argmax(1)


def main(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    info = INFO[args.dataset]
    model = build(len(info["label"]), info["n_channels"], args.size).to(device).eval()
    model.load_state_dict(torch.load(f"data/models/{args.dataset}_{args.size}.pt", map_location=device))
    xv, _ = arrays(args.dataset, "val", args.size)
    xt, yt = arrays(args.dataset, "test", args.size)
    reference, _ = frame(model, xv, device)
    rng = np.random.default_rng(args.seed)
    out = Path("results/drift")
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for kind in ("clean", "noise", "blur", "low_contrast"):
        current, pred = frame(model, corrupt(xt, kind, rng), device)
        snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
        snapshot.save_html(str(out / f"{args.dataset}_{args.size}_{kind}.html"))
        columns = {}
        for m in snapshot.dict()["metrics"]:
            cfg = m["config"]
            if not cfg["type"].endswith("ValueDrift"):
                continue
            # Evidently picks the test by sample size: a p-value (drift if below) or a distance (drift if above)
            value, threshold = float(m["value"]), cfg["threshold"]
            drift = value < threshold if "p_value" in cfg["method"] else value >= threshold
            columns[cfg["column"]] = {"method": cfg["method"], "value": round(value, 4), "drift": drift}
        summary[kind] = {"accuracy": float((pred == yt).mean()), "drift": columns,
                         "drifted_columns": [c for c, v in columns.items() if v["drift"]]}
        print(f"  {kind:13} accuracy {summary[kind]['accuracy']:.3f}  drifted: {summary[kind]['drifted_columns']}")
    (out / f"{args.dataset}_{args.size}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu to leave the GPU to another job")
    main(parser.parse_args())
