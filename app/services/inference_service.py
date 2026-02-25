"""
app/services/inference_service.py

Loads CNN + XGBoost ONCE at server startup; exposes run_inference().
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
import xgboost as xgb
from PIL import Image

logger = logging.getLogger(__name__)

# ─── Module-level singletons (loaded once at startup) ────────────────────────
_cnn_model: torchvision.models.ResNet | None = None
_xgb_model: xgb.Booster | None = None
_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

MODEL_DIR = Path("models")


def load_models() -> None:
    """Call once at FastAPI startup event."""
    global _cnn_model, _xgb_model

    logger.info("Loading CNN model …")
    cnn = torchvision.models.resnet18()
    cnn.fc = torch.nn.Linear(cnn.fc.in_features, 2)
    cnn.load_state_dict(
        torch.load(str(MODEL_DIR / "cnn_model.pth"), map_location="cpu")
    )
    cnn.eval()
    _cnn_model = cnn

    logger.info("Loading XGBoost model …")
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(MODEL_DIR / "xgb_dom.json"))
    _xgb_model = xgb_model

    logger.info("✅ Models loaded successfully.")


def run_inference(image_path: str, dom_json_path: str) -> dict:
    """
    Run CNN + XGBoost inference.

    Returns:
        {
            "prob_issue": float,      # 0-1  (CNN softmax, class 1)
            "usability_score": float, # XGB predicted usability
            "severity": float,        # fused severity index
        }
    """
    if _cnn_model is None or _xgb_model is None:
        raise RuntimeError("Models not loaded — call load_models() first.")

    # ── CNN ──────────────────────────────────────────────────────────────────
    img = Image.open(image_path).convert("RGB")
    x = _transform(img).unsqueeze(0)
    with torch.no_grad():
        out = _cnn_model(x)
        prob_issue = float(torch.softmax(out, dim=1)[0, 1].item())

    # ── XGBoost ──────────────────────────────────────────────────────────────
    with open(dom_json_path, "r", encoding="utf-8") as f:
        dom = json.load(f)

    feats = [
        dom.get("interactive_count", 0),
        dom.get("images_count", 0),
        dom.get("imagesWithoutAlt", 0),
        dom.get("accessibility", {}).get("linksWithoutText", 0),
        dom.get("textDensity", {}).get("wordCount", 0),
    ]
    dom_pred = float(_xgb_model.predict(xgb.DMatrix([feats]))[0])

    severity = prob_issue * (1.0 / (dom_pred + 1e-6))

    return {
        "prob_issue": round(prob_issue, 4),
        "usability_score": round(dom_pred, 4),
        "severity": round(severity, 4),
    }
