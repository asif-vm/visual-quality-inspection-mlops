from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.model import DefectCNN


ARTIFACT = Path(__file__).parent / "artifacts" / "model.pt"
app = FastAPI(title="Visual Quality Inspection API", version="1.0.0")
model = DefectCNN()
ready = False
if ARTIFACT.exists():
    bundle = torch.load(ARTIFACT, map_location="cpu", weights_only=True)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    ready = True


@app.get("/health")
def health():
    return {"model_ready": ready}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    if not ready:
        raise HTTPException(503, "Run train.py first")
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(415, "Upload a PNG or JPEG image")
    try:
        image = Image.open(BytesIO(await file.read())).convert("L").resize((32, 32))
    except Exception as exc:
        raise HTTPException(400, "Invalid image") from exc
    values = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.tensor(values).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    defect_probability = float(probabilities[1])
    return {
        "classification": "defect" if defect_probability >= 0.5 else "normal",
        "defect_probability": round(defect_probability, 4),
        "review_required": 0.4 <= defect_probability <= 0.6,
    }

