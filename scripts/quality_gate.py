"""Quality gate: the classifier is still good enough, and the methods the viewer may call "trusted" still pass the
sanity check. Logs both to MLflow and fails the build otherwise.

usage: uv run --group mlops scripts/quality_gate.py --size 128
"""
import argparse
import json
import sys
from pathlib import Path

import mlflow


def check(dataset: str, benchmark: dict, verdict: dict, limits: dict) -> list[str]:
    failures = []
    auc = benchmark["auc"]
    if auc < limits["auc"]["min"]:
        failures.append(f"{dataset}: test AUC {auc:.3f} below {limits['auc']['min']}")
    for method in limits["must_pass_sanity"]:
        if not verdict.get(method, {}).get("passes_sanity", False):
            failures.append(f"{dataset}: {method} no longer passes the sanity check")
    return failures


def main(args):
    limits = json.loads(args.limits.read_text())
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("whitebox-quality-gate")
    failures = []
    for dataset, lim in limits.items():
        benchmark = json.loads(Path(f"results/benchmark/{dataset}_{args.size}.json").read_text())
        verdict = json.loads(Path(f"results/sanity/{dataset}_{args.size}_verdict.json").read_text())
        found = check(dataset, benchmark, verdict, lim)
        failures += found
        with mlflow.start_run(run_name=f"{dataset}_{args.size}"):
            mlflow.log_metrics({"test_acc": benchmark["acc"], "test_auc": benchmark["auc"],
                                **{f"sanity_{m}": v["similarity_after_top_randomized"] for m, v in verdict.items()}})
            mlflow.set_tag("gate", "failed" if found else "passed")
    if failures:
        print("quality gate FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("quality gate passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--limits", type=Path, default=Path("quality_gate.json"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    main(parser.parse_args())
