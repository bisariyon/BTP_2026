# Check what's ACTUALLY in your DOM JSON files

import pandas as pd
import json
from pathlib import Path

meta = pd.read_csv("data/labelled/metadata.csv")
print(f"Loaded {len(meta)} pages\n")

# Check first 3 DOM files
for idx in range(min(3, len(meta))):
    row = meta.iloc[idx]
    dom_path = row["dom_json"]
    
    print(f"\n{'='*80}")
    print(f"Page {idx+1}: {dom_path}")
    print(f"{'='*80}")
    
    p = Path(dom_path)
    
    # Check if file exists
    if not p.exists():
        print(f"❌ FILE DOES NOT EXIST!")
        continue
    
    # Check file size
    file_size = p.stat().st_size
    print(f"File size: {file_size} bytes")
    
    if file_size == 0:
        print(f"❌ FILE IS EMPTY!")
        continue
    
    # Try to load it
    try:
        with p.open("r", encoding="utf-8") as f:
            dom = json.load(f)
        
        print(f"\n✓ Successfully loaded JSON")
        print(f"Top-level keys: {list(dom.keys())[:10]}")
        
        # Show structure
        print(f"\nJSON structure (first 500 chars):")
        print(json.dumps(dom, indent=2)[:500])
        
    except Exception as e:
        print(f"❌ ERROR loading JSON: {e}")