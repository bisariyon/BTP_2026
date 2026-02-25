# models/inference.py
import torch
import torchvision
from torch import nn
from PIL import Image
import json
import xgboost as xgb
import pandas as pd
from pathlib import Path

# CNN setup (must match train_cnn.py)
cnn_model = torchvision.models.resnet18()
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
cnn_model.load_state_dict(torch.load("models/cnn_model.pth", map_location="cpu"))
cnn_model.eval()

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
])

# XGBoost setup
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgb_dom.json")


def predict(image_path: str, dom_json: str):
    # CNN prediction
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = cnn_model(x)
        prob_issue = torch.softmax(out, dim=1)[0, 1].item()

    # XGB prediction
    p = Path(dom_json)
    with p.open("r", encoding="utf-8") as f:
        dom = json.load(f)

    feats = {
        "interactive_count": dom.get("interactive_count", 0),
        "images_count": dom.get("images_count", 0),
        "imagesWithoutAlt": dom.get("imagesWithoutAlt", 0),
        "linksWithoutText": dom.get("accessibility", {}).get("linksWithoutText", 0),
        "wordCount": dom.get("textDensity", {}).get("wordCount", 0),
    }

    dom_pred = float(xgb_model.predict(xgb.DMatrix([list(feats.values())]))[0])

    # Combined severity index
    severity = prob_issue * (1.0 / (dom_pred + 1e-6))

    return {
        "prob_issue": prob_issue,
        "usability_score": dom_pred,
        "severity": severity,
    }


if __name__ == "__main__":
    meta = pd.read_csv("data/labelled/metadata.csv").head(3)
    for _, row in meta.iterrows():
        res = predict(row["image_path"], row["dom_json"])
        print(row["image_path"], res)
