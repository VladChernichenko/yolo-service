"""
YOLO Food Segmentation Microservice
Wraps best.pt (YOLO11n-seg fine-tuned on FoodSeg154) as a REST API.
Spring Boot no longer calls this service. POST /analyze returns detected food names only.

Start: uvicorn main:app --host 0.0.0.0 --port 8001
"""

import base64
import logging
import os
from pathlib import Path
from typing import List

# -- Compatibility shim --------------------------------------------------------
# best.pt was saved with an ultralytics version that included BCEDiceLoss.
# Newer versions removed it. We only need it to unpickle the checkpoint -
# it is never called during inference.
import torch.nn as _nn
import ultralytics.utils.loss as _loss

if not hasattr(_loss, "BCEDiceLoss"):
    class _BCEDiceLoss(_nn.Module):
        """Stub - only needed to unpickle checkpoints trained with older ultralytics."""
        def forward(self, pred, target):
            raise NotImplementedError("BCEDiceLoss stub - inference only")
    _loss.BCEDiceLoss = _BCEDiceLoss

if not hasattr(_loss, "MultiChannelDiceLoss"):
    class _MultiChannelDiceLoss(_nn.Module):
        """Stub - only needed to unpickle checkpoints trained with older ultralytics."""
        def forward(self, pred, target):
            raise NotImplementedError("MultiChannelDiceLoss stub - inference only")
    _loss.MultiChannelDiceLoss = _MultiChannelDiceLoss

# -----------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO
import io

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# -- Paths ---------------------------------------------------------------------
SERVICE_DIR   = Path(__file__).parent
PROJECT_ROOT  = SERVICE_DIR.parent
# MODEL_PATH can be overridden via env var for Docker deployments
MODEL_PATH    = Path(os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "best.pt")))
# -- Load model at startup -----------------------------------------------------
log.info(f"Loading YOLO model from {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

# -- FastAPI -------------------------------------------------------------------
app = FastAPI(title="YOLO Food Vision Service", version="1.0.0")


class AnalyzeRequest(BaseModel):
    image: str            # base64-encoded image bytes
    mime_type: str = "image/jpeg"
    confidence_threshold: float = 0.35


class AnalyzeResponse(BaseModel):
    detected_foods: List[str]
    confidences: List[float]
    nutrition: dict


@app.get("/health")
def health():
    return {"status": "ok", "model": str(MODEL_PATH), "classes": len(model.names)}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    # Decode image
    try:
        img_bytes = base64.b64decode(request.image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Run YOLO inference
    results = model(img, conf=request.confidence_threshold, verbose=False)

    # Collect detections - deduplicate by class name, keep highest confidence
    best: dict[str, float] = {}
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id   = int(box.cls[0])
            conf     = float(box.conf[0])
            name     = model.names[cls_id]
            if name not in best or conf > best[name]:
                best[name] = conf

    detected_foods = list(best.keys())
    confidences    = [best[f] for f in detected_foods]

    log.info(f"Detected {len(detected_foods)} food(s): {detected_foods}")

    return AnalyzeResponse(
        detected_foods=detected_foods,
        confidences=confidences,
        nutrition={},
    )
