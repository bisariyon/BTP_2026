# Check if model is overfitting or if there's data leakage

import pandas as pd
import json
import xgboost as xgb
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print("OVERFITTING & DATA LEAKAGE CHECK")
print("="*90)

# Load data
meta = pd.read_csv("data/labelled/metadata.csv")
print(f"\n✓ Loaded {len(meta)} pages")

def extract_features(dom_json_path: str):
    try:
        p = Path(dom_json_path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        dom = data.get("dom", {})
        return {
            "interactive_count": dom.get("interactive_count", 0),
            "images_count": dom.get("images_count", 0),
            "imagesWithoutAlt": dom.get("imagesWithoutAlt", 0),
            "linksWithoutText": dom.get("accessibility", {}).get("linksWithoutText", 0),
            "wordCount": dom.get("textDensity", {}).get("wordCount", 0),
        }
    except:
        return None

X_orig = []
y = []

for _, row in meta.iterrows():
    feats = extract_features(row["dom_json"])
    if feats:
        X_orig.append(list(feats.values()))
        y.append(float(row["usability_score"]))

X_orig = np.array(X_orig)
y = np.array(y)

print(f"✓ Extracted 5 features from {len(X_orig)} pages")

# Engineer features
X_engineered = []

for i in range(len(X_orig)):
    interactive_count = X_orig[i, 0]
    images_count = X_orig[i, 1]
    imagesWithoutAlt = X_orig[i, 2]
    linksWithoutText = X_orig[i, 3]
    wordCount = X_orig[i, 4]
    
    alt_text_ratio = (images_count - imagesWithoutAlt) / max(images_count, 1)
    link_text_ratio = (interactive_count - linksWithoutText) / max(interactive_count, 1)
    missing_alt_ratio = imagesWithoutAlt / max(images_count, 1)
    missing_link_ratio = linksWithoutText / max(interactive_count, 1)
    accessibility_score = ((interactive_count - linksWithoutText) + (images_count - imagesWithoutAlt)) / max(interactive_count + images_count, 1)
    total_elements = interactive_count + images_count
    content_completeness = min(1.0, wordCount / 500)
    
    features = {
        'interactive_count': interactive_count,
        'images_count': images_count,
        'imagesWithoutAlt': imagesWithoutAlt,
        'linksWithoutText': linksWithoutText,
        'wordCount': wordCount,
        'alt_text_ratio': alt_text_ratio,
        'link_text_ratio': link_text_ratio,
        'missing_alt_ratio': missing_alt_ratio,
        'missing_link_ratio': missing_link_ratio,
        'accessibility_score': accessibility_score,
        'total_elements': total_elements,
        'element_to_word_ratio': max(interactive_count + images_count, 1) / max(wordCount, 1),
        'interactive_density': interactive_count / max(interactive_count + images_count, 1),
        'image_density': images_count / max(interactive_count + images_count, 1),
        'element_richness': (interactive_count + images_count) / max(1, wordCount / 100),
        'content_to_elements': wordCount / max(interactive_count + images_count, 1),
        'readability_proxy': (wordCount / max(interactive_count, 1)) * 0.5,
        'information_density': min(1.0, (wordCount / 5000)),
        'content_completeness': content_completeness,
        'media_integration': (images_count / max(interactive_count + images_count, 1)) if (interactive_count + images_count) > 0 else 0,
        'accessibility_completeness': (alt_text_ratio + link_text_ratio) / 2,
        'alt_text_missing_severity': min(1.0, imagesWithoutAlt / max(images_count, 1)),
        'link_text_missing_severity': min(1.0, linksWithoutText / max(interactive_count, 1)),
        'accessibility_issues_count': imagesWithoutAlt + linksWithoutText,
        'accessibility_issue_ratio': (imagesWithoutAlt + linksWithoutText) / max(interactive_count + images_count, 1),
        'has_interactive_elements': 1 if interactive_count > 0 else 0,
        'interactive_element_count': interactive_count,
        'interactive_completeness': (interactive_count - linksWithoutText) / max(interactive_count, 1),
        'interactive_quality': link_text_ratio,
        'button_link_ratio': interactive_count / max(1, wordCount),
        'has_images': 1 if images_count > 0 else 0,
        'image_count': images_count,
        'image_completeness': (images_count - imagesWithoutAlt) / max(images_count, 1),
        'image_quality': alt_text_ratio,
        'image_to_word_ratio': images_count / max(1, wordCount / 100),
        'page_complexity': (interactive_count + images_count) / 100,
        'interaction_level': interactive_count / 10,
        'media_richness': (images_count + interactive_count) / 20,
        'feature_richness': min(3.0, (interactive_count + images_count) / 10),
        'element_variety': 1 if (interactive_count > 0 and images_count > 0) else 0.5,
        'overall_quality': (accessibility_score * 0.4 + content_completeness * 0.3 + (1 if interactive_count + images_count > 0 else 0) * 0.3),
        'content_quality_index': min(1.0, (wordCount / 300) * (1 - (imagesWithoutAlt + linksWithoutText) / max(interactive_count + images_count, 1))),
        'usability_score_proxy': (link_text_ratio * 0.3 + alt_text_ratio * 0.3 + content_completeness * 0.4),
        'element_quality_average': (link_text_ratio + alt_text_ratio) / 2,
        'overall_health': min(1.0, (wordCount / 200) * (1 - (imagesWithoutAlt + linksWithoutText) / max(interactive_count + images_count, 50))),
    }
    
    X_engineered.append(list(features.values()))

X_engineered = np.array(X_engineered)
feature_names = list(features.keys())

print(f"✓ Engineered {len(feature_names)} features")

# ============================================================================
# TEST 1: TRAIN vs VALIDATION OVERFITTING
# ============================================================================
print(f"\n" + "="*90)
print("TEST 1: TRAIN vs VALIDATION (OVERFITTING CHECK)")
print("="*90)

X_train, X_val, y_train, y_val = train_test_split(
    X_engineered, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dval = xgb.DMatrix(X_val_scaled, label=y_val)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

evals = [(dtrain, 'train'), (dval, 'validation')]
bst = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)

y_pred_train = bst.predict(dtrain)
y_pred_val = bst.predict(dval)

r2_train = r2_score(y_train, y_pred_train)
r2_val = r2_score(y_val, y_pred_val)

accuracy_train = accuracy_score((y_train >= 0.5).astype(int), (y_pred_train >= 0.5).astype(int))
accuracy_val = accuracy_score((y_val >= 0.5).astype(int), (y_pred_val >= 0.5).astype(int))

print(f"\nR² SCORE:")
print(f"  Train: {r2_train:.4f}")
print(f"  Val:   {r2_val:.4f}")
print(f"  Gap:   {abs(r2_train - r2_val):.4f} {'⚠️ OVERFITTING' if abs(r2_train - r2_val) > 0.1 else '✅ OK'}")

print(f"\nACCURACY:")
print(f"  Train: {accuracy_train:.4f}")
print(f"  Val:   {accuracy_val:.4f}")
print(f"  Gap:   {abs(accuracy_train - accuracy_val):.4f} {'⚠️ OVERFITTING' if abs(accuracy_train - accuracy_val) > 0.05 else '✅ OK'}")

# ============================================================================
# TEST 2: CROSS-VALIDATION (MORE ROBUST)
# ============================================================================
print(f"\n" + "="*90)
print("TEST 2: K-FOLD CROSS-VALIDATION (5 FOLDS)")
print("="*90)

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_scores = []
cv_accuracy_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_engineered), 1):
    X_train_cv = X_engineered[train_idx]
    X_val_cv = X_engineered[val_idx]
    y_train_cv = y[train_idx]
    y_val_cv = y[val_idx]
    
    scaler_cv = StandardScaler()
    X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
    X_val_cv_scaled = scaler_cv.transform(X_val_cv)
    
    dtrain_cv = xgb.DMatrix(X_train_cv_scaled, label=y_train_cv)
    dval_cv = xgb.DMatrix(X_val_cv_scaled, label=y_val_cv)
    
    bst_cv = xgb.train(params, dtrain_cv, num_boost_round=200, verbose_eval=False)
    
    y_pred_cv = bst_cv.predict(dval_cv)
    r2_cv = r2_score(y_val_cv, y_pred_cv)
    accuracy_cv = accuracy_score((y_val_cv >= 0.5).astype(int), (y_pred_cv >= 0.5).astype(int))
    
    cv_r2_scores.append(r2_cv)
    cv_accuracy_scores.append(accuracy_cv)
    
    print(f"\nFold {fold}:")
    print(f"  R²:       {r2_cv:.4f}")
    print(f"  Accuracy: {accuracy_cv:.4f}")

print(f"\nCross-Validation Results:")
print(f"  R² Mean:       {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}")
print(f"  Accuracy Mean: {np.mean(cv_accuracy_scores):.4f} ± {np.std(cv_accuracy_scores):.4f}")

# ============================================================================
# TEST 3: CHECK FOR DATA LEAKAGE
# ============================================================================
print(f"\n" + "="*90)
print("TEST 3: DATA LEAKAGE CHECK")
print("="*90)

# Check if engineered features are too correlated with target
from scipy.stats import pearsonr

correlations = []
for i, fname in enumerate(feature_names):
    corr, pval = pearsonr(X_engineered[:, i], y)
    correlations.append((fname, abs(corr)))

correlations_sorted = sorted(correlations, key=lambda x: x[1], reverse=True)

print(f"\nTop 10 feature correlations with target:")
print(f"Rank | Feature Name                      | Correlation")
print("-" * 70)
for rank, (fname, corr) in enumerate(correlations_sorted[:10], 1):
    if corr > 0.9:
        print(f"{rank:>3} | {fname:<35} | {corr:.4f} ⚠️  VERY HIGH!")
    elif corr > 0.7:
        print(f"{rank:>3} | {fname:<35} | {corr:.4f} ⚠️  HIGH")
    else:
        print(f"{rank:>3} | {fname:<35} | {corr:.4f}")

# ============================================================================
# CONCLUSION
# ============================================================================
print(f"\n" + "="*90)
print("DIAGNOSIS")
print("="*90)

mean_r2 = np.mean(cv_r2_scores)
mean_acc = np.mean(cv_accuracy_scores)

print(f"\nCross-Validation Performance (Most Reliable):")
print(f"  R²:       {mean_r2:.4f} (±{np.std(cv_r2_scores):.4f})")
print(f"  Accuracy: {mean_acc:.4f} (±{np.std(cv_accuracy_scores):.4f})")

if abs(r2_train - r2_val) > 0.1 or abs(accuracy_train - accuracy_val) > 0.05:
    print(f"\n⚠️  WARNING: Model is OVERFITTING")
    print(f"   - Train/Val gap is large")
    print(f"   - Need stronger regularization")
elif max(correlations_sorted, key=lambda x: x[1])[1] > 0.95:
    print(f"\n⚠️  WARNING: Possible DATA LEAKAGE")
    print(f"   - Features are too correlated with target")
    print(f"   - Check feature engineering")
else:
    print(f"\n✅ Model seems OK")
    print(f"   - CV performance = {mean_r2:.4f} R²")
    print(f"   - CV accuracy = {mean_acc:.4f}")
    print(f"   - This is more realistic than 0.95")

print(f"\n" + "="*90)