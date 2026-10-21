import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("quality_gate", Path(__file__).parents[1] / "scripts" / "quality_gate.py")


def test_gate_catches_a_weak_model_and_a_method_that_stopped_passing():
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    limits = {"auc": {"min": 0.9}, "must_pass_sanity": ["gradcam"]}
    assert gate.check("x", {"auc": 0.95}, {"gradcam": {"passes_sanity": True}}, limits) == []
    failures = gate.check("x", {"auc": 0.80}, {"gradcam": {"passes_sanity": False}}, limits)
    assert len(failures) == 2
