"""
YOLO Food Segmentation Microservice
Wraps best.pt (YOLO11n-seg fine-tuned on FoodSeg154) as a REST API.
Spring Boot calls POST /analyze with a base64 image and gets nutrition back.

Start: uvicorn main:app --host 0.0.0.0 --port 8001
"""

import base64
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Optional, List

# ── Compatibility shim ────────────────────────────────────────────────────────
# best.pt was saved with an ultralytics version that included BCEDiceLoss.
# Newer versions removed it. We only need it to unpickle the checkpoint —
# it is never called during inference.
import torch.nn as _nn
import ultralytics.utils.loss as _loss

if not hasattr(_loss, "BCEDiceLoss"):
    class _BCEDiceLoss(_nn.Module):
        """Stub — only needed to unpickle checkpoints trained with older ultralytics."""
        def forward(self, pred, target):
            raise NotImplementedError("BCEDiceLoss stub — inference only")
    _loss.BCEDiceLoss = _BCEDiceLoss

if not hasattr(_loss, "MultiChannelDiceLoss"):
    class _MultiChannelDiceLoss(_nn.Module):
        """Stub — only needed to unpickle checkpoints trained with older ultralytics."""
        def forward(self, pred, target):
            raise NotImplementedError("MultiChannelDiceLoss stub — inference only")
    _loss.MultiChannelDiceLoss = _MultiChannelDiceLoss

# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO
import io

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
SERVICE_DIR   = Path(__file__).parent
PROJECT_ROOT  = SERVICE_DIR.parent
# MODEL_PATH can be overridden via env var for Docker deployments
MODEL_PATH    = Path(os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "best.pt")))
NUTRITION_MAP_PATH = SERVICE_DIR / "nutrition_map.json"

# ── Load model & nutrition map at startup ─────────────────────────────────────
log.info(f"Loading YOLO model from {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

with open(NUTRITION_MAP_PATH) as f:
    NUTRITION_MAP: dict = json.load(f)

log.info(f"Loaded nutrition map with {len(NUTRITION_MAP)} entries")

# ── FastAPI ───────────────────────────────────────────────────────────────────
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

    # Collect detections — deduplicate by class name, keep highest confidence
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

    nutrition = _aggregate_nutrition(detected_foods)
    return AnalyzeResponse(
        detected_foods=detected_foods,
        confidences=confidences,
        nutrition=nutrition,
    )


# ── Nutrition aggregation ─────────────────────────────────────────────────────

# Per-protein-type FPU kinetic constants (Warsaw Protocol + clinical literature)
_PROTEIN_FPU: dict[str, dict] = {
    "ANIMAL":   {"gluconeogenic_fraction": 0.50, "onset_min": 90},
    "CASEIN":   {"gluconeogenic_fraction": 0.50, "onset_min": 120},
    "WHEY":     {"gluconeogenic_fraction": 0.50, "onset_min": 60},
    "PLANT":    {"gluconeogenic_fraction": 0.35, "onset_min": 100},
    "COLLAGEN": {"gluconeogenic_fraction": 0.10, "onset_min": 90},
    "NONE":     {"gluconeogenic_fraction": 0.00, "onset_min": 0},
}

# MCT fats bypass FPU-related gastric delay (oxidised directly, no portal glucose)
_FAT_LCT_FRACTION: dict[str, float] = {"MCT": 0.0, "LCT": 1.0, "NONE": 0.0}


def _lookup(food_name: str) -> Optional[dict]:
    """Exact match first, then substring fallback."""
    if food_name in NUTRITION_MAP:
        return NUTRITION_MAP[food_name]
    # try partial match (e.g. "French fries" vs "french fries")
    lower = food_name.lower()
    for key, val in NUTRITION_MAP.items():
        if lower == key.lower():
            return val
    for key, val in NUTRITION_MAP.items():
        if lower in key.lower() or key.lower() in lower:
            return val
    return None


def _aggregate_nutrition(foods: List[str]) -> dict:
    if not foods:
        return {
            "totalCarbs": 0.0, "fiber": 0.0, "protein": 0.0, "fat": 0.0,
            "estimatedGi": 0.0, "glycemicLoad": 0.0, "absorptionSpeedClass": "DEFAULT",
            "lctFatG": 0.0, "fpuOnsetMin": 90, "gluconeogenicFraction": 0.50,
        }

    total_carbs        = 0.0
    total_fiber        = 0.0
    total_protein      = 0.0
    total_fat          = 0.0
    gi_x_carbs         = 0.0   # for weighted-average GI
    speeds: list[str]  = []

    total_lct_fat_g    = 0.0
    total_protein_kcal = 0.0
    weighted_onset     = 0.0
    weighted_gluconeo  = 0.0

    for food in foods:
        data = _lookup(food)
        if data is None:  # type: ignore
            log.warning(f"No nutrition data for '{food}' — skipping")
            continue

        factor    = data.get("serving_g", 100) / 100.0
        carbs     = data.get("carbs",   0) * factor
        protein_g = data.get("protein", 0) * factor
        fat_g     = data.get("fat",     0) * factor
        gi        = data.get("gi", 0)

        total_carbs   += carbs
        total_fiber   += data.get("fiber", 0) * factor
        total_protein += protein_g
        total_fat     += fat_g
        gi_x_carbs    += gi * carbs
        speeds.append(data.get("absorption", "DEFAULT"))

        # FPU type-aware aggregation — weighted by protein kcal contribution
        fat_type     = data.get("fat_type",     "LCT")
        protein_type = data.get("protein_type", "ANIMAL")

        lct_frac = _FAT_LCT_FRACTION.get(fat_type, 1.0)
        total_lct_fat_g += fat_g * lct_frac

        p_kcal = protein_g * 4.0
        total_protein_kcal += p_kcal
        fpu = _PROTEIN_FPU.get(protein_type, _PROTEIN_FPU["ANIMAL"])
        weighted_onset   += p_kcal * fpu["onset_min"]
        weighted_gluconeo += p_kcal * fpu["gluconeogenic_fraction"]

    avg_gi = (gi_x_carbs / total_carbs) if total_carbs > 0 else 0.0
    gl     = (avg_gi * total_carbs) / 100.0

    dominant_speed = Counter(speeds).most_common(1)[0][0] if speeds else "DEFAULT"

    fpu_onset_min      = round(weighted_onset    / total_protein_kcal) if total_protein_kcal > 0 else 90
    gluconeogenic_frac = (weighted_gluconeo / total_protein_kcal)      if total_protein_kcal > 0 else 0.50

    return {
        "totalCarbs":            round(total_carbs,       1),
        "fiber":                 round(total_fiber,       1),
        "protein":               round(total_protein,     1),
        "fat":                   round(total_fat,         1),
        "estimatedGi":           round(avg_gi,            1),
        "glycemicLoad":          round(gl,                1),
        "absorptionSpeedClass":  dominant_speed,
        "lctFatG":               round(total_lct_fat_g,   1),
        "fpuOnsetMin":           fpu_onset_min,
        "gluconeogenicFraction": round(gluconeogenic_frac, 3),
    }
