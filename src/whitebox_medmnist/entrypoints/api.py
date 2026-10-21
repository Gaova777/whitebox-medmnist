"""HTTP API for the viewer. Educational demo: not a diagnostic tool.

GET  /datasets                     what can be loaded
GET  /samples/{dataset}            a few test images to try (as PNG data URLs) with their true label
POST /predict/{dataset}            an image -> class probabilities
POST /explain/{dataset}            an image -> one heat map per method, each with its trust badge
GET  /drift/{dataset}              the drift test summary
"""
import base64
import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from whitebox_medmnist.domain.trust import badge, top_k_overlap

SIZE = int(os.environ.get("WBM_SIZE", "128"))
DATASETS = ["pneumoniamnist", "bloodmnist"]
NOTICE = "Educational demo of explainability methods. Not a medical device; not for diagnosis."

app = FastAPI(title="whitebox-medmnist")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:4200"], allow_methods=["*"], allow_headers=["*"])


@lru_cache
def model(dataset: str):
    from whitebox_medmnist.adapters.medmnist_data import INFO
    from whitebox_medmnist.application.model import build

    info = INFO[dataset]
    m = build(len(info["label"]), info["n_channels"], SIZE)
    m.load_state_dict(torch.load(f"data/models/{dataset}_{SIZE}.pt", map_location="cpu"))
    return m.eval(), info["n_channels"]


@lru_cache
def verdict(dataset: str) -> dict:
    path = Path(f"results/sanity/{dataset}_{SIZE}_verdict.json")
    return json.loads(path.read_text()) if path.exists() else {}


def check(dataset: str):
    if dataset not in DATASETS:
        raise HTTPException(404, f"unknown dataset {dataset}")


def to_tensor(raw: bytes, channels: int) -> torch.Tensor:
    try:
        img = Image.open(io.BytesIO(raw)).convert("L" if channels == 1 else "RGB").resize((SIZE, SIZE))
    except Exception as e:
        raise HTTPException(400, f"not an image: {e}") from e
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = a[None] if a.ndim == 2 else a.transpose(2, 0, 1)
    return torch.from_numpy(a)[None]


def png(a: np.ndarray) -> str:
    a = (255 * (a - a.min()) / (np.ptp(a) or 1)).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/datasets")
def datasets():
    from whitebox_medmnist.adapters.medmnist_data import classes

    return {"notice": NOTICE, "size": SIZE, "datasets": {d: classes(d) for d in DATASETS}}


@app.get("/samples/{dataset}")
def samples(dataset: str, n: int = 12):
    check(dataset)
    from whitebox_medmnist.adapters.medmnist_data import arrays, classes

    x, y = arrays(dataset, "test", SIZE)
    idx = np.random.default_rng(0).choice(len(x), n, replace=False)
    names = classes(dataset)
    return [{"image": png(x[i].transpose(1, 2, 0).squeeze()), "label": names[y[i]]} for i in idx]


@app.post("/predict/{dataset}")
async def predict(dataset: str, image: Annotated[UploadFile, File()]):
    check(dataset)
    from whitebox_medmnist.adapters.medmnist_data import classes

    m, channels = model(dataset)
    with torch.no_grad():
        probs = m(to_tensor(await image.read(), channels)).softmax(-1)[0].tolist()
    return {"notice": NOTICE, "probabilities": dict(zip(classes(dataset), probs))}


@app.post("/explain/{dataset}")
async def explain_image(dataset: str, image: Annotated[UploadFile, File()]):
    check(dataset)
    from whitebox_medmnist.adapters.medmnist_data import classes
    from whitebox_medmnist.application.saliency import METHODS, explain

    m, channels = model(dataset)
    x = to_tensor(await image.read(), channels)
    with torch.no_grad():
        probs = m(x).softmax(-1)[0]
    target = probs.argmax().view(1)
    gen = torch.Generator().manual_seed(0)
    noisy = [(x + 0.02 * torch.randn(x.shape, generator=gen)).clamp(0, 1) for _ in range(3)]
    k = int(0.05 * SIZE * SIZE)
    out = []
    for method in METHODS:
        heat = explain(m, x, target, method)[0].numpy()
        copies = [explain(m, n, target, method)[0].numpy().ravel().tolist() for n in noisy]
        stability = float(np.mean([top_k_overlap(copies[i], copies[j], k) for i in range(3) for j in range(i + 1, 3)]))
        passed = verdict(dataset).get(method, {}).get("passes_sanity", False)
        out.append({"method": method, "map": png(heat), "stability": round(stability, 3), "badge": badge(passed, stability)})
    return {"notice": NOTICE, "prediction": classes(dataset)[int(target)], "confidence": float(probs.max()),
            "explanations": out}


@app.get("/drift/{dataset}")
def drift(dataset: str):
    check(dataset)
    path = Path(f"results/drift/{dataset}_{SIZE}_summary.json")
    return json.loads(path.read_text()) if path.exists() else {}
