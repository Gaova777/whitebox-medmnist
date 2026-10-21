import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from whitebox_medmnist.application.model import build
from whitebox_medmnist.entrypoints import api


def image():
    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(0).random((28, 28)) * 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_explain_gives_six_badged_maps_and_the_notice(monkeypatch):
    monkeypatch.setattr(api, "SIZE", 28)
    m = build(2, 1, 28).eval()
    monkeypatch.setattr(api, "model", lambda d: (m, 1))
    monkeypatch.setattr(api, "verdict", lambda d: {"gradient": {"passes_sanity": True}})
    c = TestClient(api.app)
    r = c.post("/explain/pneumoniamnist", files={"image": ("x.png", image(), "image/png")}).json()
    assert "Not a medical device" in r["notice"] and len(r["explanations"]) == 6
    levels = {e["method"]: e["badge"]["level"] for e in r["explanations"]}
    assert levels["guided_backprop"] == "not an explanation" and levels["gradient"] in ("trusted", "unstable")
    assert c.post("/explain/chestmnist", files={"image": ("x.png", image(), "image/png")}).status_code == 404
