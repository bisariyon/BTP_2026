# FASTEST WAY: Extract 40+ features from your existing 5 features
# Run: python quick_xgboost_fix.py

import pandas as pd
import json
import xgboost as xgb
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print("XGBOOST QUICK FIX - ENGINEERED FEATURES FROM EXISTING DATA")
print("="*90)

# Load data
meta = pd.read_csv("data/labelled/metadata.csv")

def extract_features(dom_json_path: str):
    """Extract original 5 features - CORRECTED PATH"""
    try:
        p = Path(dom_json_path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # The features are nested under "dom" key!
        dom = data.get("dom", {})
        
        return {
            "interactive_count": dom.get("interactive_count", 0),
            "images_count": dom.get("images_count", 0),
            "imagesWithoutAlt": dom.get("imagesWithoutAlt", 0),
            "linksWithoutText": dom.get("accessibility", {}).get("linksWithoutText", 0),
            "wordCount": dom.get("textDensity", {}).get("wordCount", 0),
        }
    except Exception as e:
        print(f"Error loading {dom_json_path}: {e}")
        return None

# Extract original 5 features
X_orig = []
y = []

for idx, (_, row) in enumerate(meta.iterrows()):
    feats = extract_features(row["dom_json"])
    if feats:
        X_orig.append(list(feats.values()))
        y.append(float(row["usability_score"]))

X_orig = np.array(X_orig)
y = np.array(y)


X_engineered = []

for i in range(len(X_orig)):
    interactive_count = X_orig[i, 0]
    images_count = X_orig[i, 1]
    imagesWithoutAlt = X_orig[i, 2]
    linksWithoutText = X_orig[i, 3]
    wordCount = X_orig[i, 4]
    
    # Calculate ratios FIRST (before using them)
    alt_text_ratio = (images_count - imagesWithoutAlt) / max(images_count, 1)
    link_text_ratio = (interactive_count - linksWithoutText) / max(interactive_count, 1)
    missing_alt_ratio = imagesWithoutAlt / max(images_count, 1)
    missing_link_ratio = linksWithoutText / max(interactive_count, 1)
    accessibility_score = ((interactive_count - linksWithoutText) + (images_count - imagesWithoutAlt)) / max(interactive_count + images_count, 1)
    total_elements = interactive_count + images_count
    content_completeness = min(1.0, wordCount / 500)
    
    features = {
        # ========== ORIGINAL 5 ==========
        'interactive_count': interactive_count,
        'images_count': images_count,
        'imagesWithoutAlt': imagesWithoutAlt,
        'linksWithoutText': linksWithoutText,
        'wordCount': wordCount,
        
        # ========== RATIO FEATURES (5) ==========
        'alt_text_ratio': alt_text_ratio,
        'link_text_ratio': link_text_ratio,
        'missing_alt_ratio': missing_alt_ratio,
        'missing_link_ratio': missing_link_ratio,
        'accessibility_score': accessibility_score,
        
        # ========== ELEMENT DENSITY FEATURES (5) ==========
        'total_elements': total_elements,
        'element_to_word_ratio': max(interactive_count + images_count, 1) / max(wordCount, 1),
        'interactive_density': interactive_count / max(interactive_count + images_count, 1),
        'image_density': images_count / max(interactive_count + images_count, 1),
        'element_richness': (interactive_count + images_count) / max(1, wordCount / 100),
        
        # ========== CONTENT QUALITY APPROXIMATIONS (5) ==========
        'content_to_elements': wordCount / max(interactive_count + images_count, 1),
        'readability_proxy': (wordCount / max(interactive_count, 1)) * 0.5,
        'information_density': min(1.0, (wordCount / 5000)),
        'content_completeness': content_completeness,
        'media_integration': (images_count / max(interactive_count + images_count, 1)) if (interactive_count + images_count) > 0 else 0,
        
        # ========== ACCESSIBILITY PROXIES (5) ==========
        'accessibility_completeness': (alt_text_ratio + link_text_ratio) / 2,
        'alt_text_missing_severity': min(1.0, imagesWithoutAlt / max(images_count, 1)),
        'link_text_missing_severity': min(1.0, linksWithoutText / max(interactive_count, 1)),
        'accessibility_issues_count': imagesWithoutAlt + linksWithoutText,
        'accessibility_issue_ratio': (imagesWithoutAlt + linksWithoutText) / max(interactive_count + images_count, 1),
        
        # ========== INTERACTIVE ELEMENT FEATURES (5) ==========
        'has_interactive_elements': 1 if interactive_count > 0 else 0,
        'interactive_element_count': interactive_count,
        'interactive_completeness': (interactive_count - linksWithoutText) / max(interactive_count, 1),
        'interactive_quality': link_text_ratio,
        'button_link_ratio': interactive_count / max(1, wordCount),
        
        # ========== IMAGE FEATURES (5) ==========
        'has_images': 1 if images_count > 0 else 0,
        'image_count': images_count,
        'image_completeness': (images_count - imagesWithoutAlt) / max(images_count, 1),
        'image_quality': alt_text_ratio,
        'image_to_word_ratio': images_count / max(1, wordCount / 100),
        
        # ========== COMPLEXITY FEATURES (5) ==========
        'page_complexity': (interactive_count + images_count) / 100,
        'interaction_level': interactive_count / 10,
        'media_richness': (images_count + interactive_count) / 20,
        'feature_richness': min(3.0, (interactive_count + images_count) / 10),
        'element_variety': 1 if (interactive_count > 0 and images_count > 0) else 0.5,
        
        # ========== DERIVED QUALITY SCORES (5) ==========
        'overall_quality': (accessibility_score * 0.4 + content_completeness * 0.3 + (1 if interactive_count + images_count > 0 else 0) * 0.3),
        'content_quality_index': min(1.0, (wordCount / 300) * (1 - (imagesWithoutAlt + linksWithoutText) / max(interactive_count + images_count, 1))),
        'usability_score_proxy': (link_text_ratio * 0.3 + alt_text_ratio * 0.3 + content_completeness * 0.4),
        'element_quality_average': (link_text_ratio + alt_text_ratio) / 2,
        'overall_health': min(1.0, (wordCount / 200) * (1 - (imagesWithoutAlt + linksWithoutText) / max(interactive_count + images_count, 50))),
    }
    
    X_engineered.append(list(features.values()))

X_engineered = np.array(X_engineered)
feature_names = list(features.keys())

print(f"  Total features: {len(feature_names)}")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
print(f"✓ Features scaled")

# Train XGBoost
print("TRAINING XGBOOST")

dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=feature_names)
dval = xgb.DMatrix(X_val_scaled, label=y_val, feature_names=feature_names)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

bst = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
print(f"✓ Model trained")

# Save model
bst.save_model("models/xgb_dom_quick.json")
print(f"✓ Model saved")

# Evaluate

y_pred = bst.predict(dval)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

y_val_binary = (y_val >= 0.5).astype(int)
y_pred_binary = (y_pred >= 0.5).astype(int)

accuracy = accuracy_score(y_val_binary, y_pred_binary)
precision = precision_score(y_val_binary, y_pred_binary, zero_division=0)
recall = recall_score(y_val_binary, y_pred_binary, zero_division=0)
f1 = f1_score(y_val_binary, y_pred_binary, zero_division=0)
auc = roc_auc_score(y_val_binary, y_pred)


importance = bst.get_score(importance_type='weight')
sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

if sorted_imp:
    print(f"\nRank | Feature Name                      | Importance")
    print("-" * 70)
    for rank, (feat, imp) in enumerate(sorted_imp[:15], 1):
        print(f"{rank:>3} | {feat:<35} | {imp:>3}")
else:
    print("(No feature importance data)")

print("✅ DONE!")