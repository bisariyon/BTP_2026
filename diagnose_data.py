# diagnose_data.py
# Check your data quality FIRST

import pandas as pd
import json
import numpy as np
from pathlib import Path

print("="*90)
print("DATA QUALITY DIAGNOSIS")
print("="*90)

# Load metadata
meta = pd.read_csv("data/labelled/metadata.csv")
print(f"\n✓ Loaded {len(meta)} pages")

# Check usability scores
scores = meta['usability_score'].values
print(f"\nUSABILITY SCORE ANALYSIS:")
print(f"  Min:    {scores.min():.4f}")
print(f"  Max:    {scores.max():.4f}")
print(f"  Mean:   {scores.mean():.4f}")
print(f"  Median: {np.median(scores):.4f}")
print(f"  Std:    {scores.std():.4f}")

# Count unique values
unique_scores = len(np.unique(scores))
print(f"  Unique values: {unique_scores}")

if unique_scores <= 10:
    print(f"\n  ⚠️  WARNING: Only {unique_scores} unique scores!")
    print(f"  Value counts:")
    value_counts = pd.Series(scores).value_counts().sort_index()
    for val, count in value_counts.items():
        print(f"      {val:.4f}: {count} pages ({count/len(scores)*100:.1f}%)")

# Check for variance
print(f"\n{'VARIANCE ANALYSIS:'}")
if scores.std() < 0.1:
    print(f"  ❌ CRITICAL: Very low variance ({scores.std():.4f})")
    print(f"     → Model can't learn! All scores are almost identical")
elif scores.std() < 0.2:
    print(f"  ⚠️  WARNING: Low variance ({scores.std():.4f})")
    print(f"     → Model will struggle")
else:
    print(f"  ✅ Good variance ({scores.std():.4f})")

# Check class distribution (for binary classification)
print(f"\nBINARY CLASSIFICATION (threshold=0.5):")
below = (scores < 0.5).sum()
above = (scores >= 0.5).sum()
print(f"  < 0.5:  {below} pages ({below/len(scores)*100:.1f}%)")
print(f"  >= 0.5: {above} pages ({above/len(scores)*100:.1f}%)")

if above == 0 or below == 0:
    print(f"  ❌ CRITICAL: One class is MISSING!")
    print(f"     → Can't train binary classifier with no positive/negative examples")

# Check feature values
print(f"\n{'='*90}")
print("FEATURE ANALYSIS")
print("="*90)

def extract_features(dom_json_path: str):
    try:
        p = Path(dom_json_path)
        with p.open("r", encoding="utf-8") as f:
            dom = json.load(f)
        return {
            "interactive_count": dom.get("interactive_count", 0),
            "images_count": dom.get("images_count", 0),
            "imagesWithoutAlt": dom.get("imagesWithoutAlt", 0),
            "linksWithoutText": dom.get("accessibility", {}).get("linksWithoutText", 0),
            "wordCount": dom.get("textDensity", {}).get("wordCount", 0),
        }
    except:
        return None

features_list = []
for idx, (_, row) in enumerate(meta.iterrows()):
    feat = extract_features(row["dom_json"])
    if feat:
        features_list.append(feat)

if features_list:
    df_features = pd.DataFrame(features_list)
    print(f"\nExtracted {len(features_list)} feature sets")
    print(f"\nFeature statistics:")
    print(df_features.describe().round(2))
    
    # Check for constant features
    for col in df_features.columns:
        if df_features[col].std() == 0:
            print(f"\n  ⚠️  WARNING: '{col}' is CONSTANT (std=0)")
            print(f"      All values = {df_features[col].iloc[0]}")

print(f"\n{'='*90}")
print("RECOMMENDATIONS")
print("="*90)

if scores.std() < 0.1:
    print(f"""
❌ PROBLEM: Your usability_score labels have NO VARIANCE!

The model predicts all "good" (0.5+) because:
  • All scores are similar (close to mean)
  • Can't learn any pattern
  • Accuracy looks good (72%) by always predicting "good"

SOLUTION OPTIONS:
1. Check your labeling process - are all pages marked as "good"?
2. Re-examine the metadata.csv usability_score column
3. Consider different target variable:
   - Could use a different column for labels?
   - Could bin scores into 3-5 categories instead of binary?
   
Let me know what the actual scores look like and I'll help fix it!
""")
else:
    print(f"""
✅ Data variance looks OK. The issue might be:
  1. Features are not predictive of the target
  2. Target variable is different than expected
  3. Need more sophisticated model
  
Let's use better hyperparameters and techniques.
""")