"""
dom_labeler.py - Convert captured DOM metrics to usability labels

This script bridges the gap between:
  1. Captured metadata JSONs (with rich DOM data)
  2. Training dataset (with scores + issues)

Usage:
  python dom_labeler.py
  
Output:
  data/labelled/metadata.csv with:
    image_path, dom_json, usability_score, issues
"""

import os
import json
import csv
from pathlib import Path
import random
import numpy as np


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/labelled")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_dom_for_issues(dom_info, visual_metrics):
    """
    Heuristic-based analysis of captured DOM to identify UX/accessibility issues.
    
    Args:
        dom_info: dict from capture metadata['dom']
        visual_metrics: dict with 'whitespace_ratio' and 'palette'
    
    Returns:
        {
            'issues': list of issue strings (or ['none'])
            'usability_score': float between 0.0-1.0
        }
    """
    issues = []
    score = 0.0
    
    # Safe extraction with defaults
    accessibility = dom_info.get('accessibility', {})
    layout = dom_info.get('layout', {})
    text_density = dom_info.get('textDensity', {})
    interactive_count = dom_info.get('interactive_count', 0)
    images_count = dom_info.get('images_count', 0)
    images_without_alt = dom_info.get('imagesWithoutAlt', 0)
    headings = dom_info.get('headings', [])
    focusable = dom_info.get('focusable', [])
    
    whitespace_ratio = visual_metrics.get('whitespace_ratio', 0.9)
    
    # ========== ISSUE DETECTION ==========
    
    # 1. Missing alt text (accessibility)
    if images_count > 0:
        alt_text_ratio = 1.0 - (images_without_alt / images_count)
        if alt_text_ratio < 0.6:  # More than 40% missing alt
            issues.append('missing_alt_text')
            score += 0.25
    
    # 2. Links without accessible text
    links_without_text = accessibility.get('linksWithoutText', 0)
    if links_without_text > 2:
        issues.append('links_without_text')
        score += 0.15
    
    # 3. Excessive DOM elements (performance + complexity)
    total_elements = layout.get('totalElements', 0)
    if total_elements > 1200:
        issues.append('excessive_divs')
        score += 0.12
    elif total_elements > 800:
        score += 0.05  # Mild penalty
    
    # 4. Deep/nested DOM (hard to navigate)
    # Calculate average nesting from interactive elements
    interactive_elements = dom_info.get('interactive', [])
    if interactive_elements:
        avg_depth = np.mean([1 for _ in interactive_elements]) if interactive_elements else 1
        if len(interactive_elements) > 30 and total_elements > 600:
            issues.append('deep_dom')
            score += 0.10
    
    # 5. Missing heading structure
    heading_count = len(headings)
    word_count = text_density.get('wordCount', 0)
    if heading_count < 2 and word_count > 250:
        issues.append('no_headings')
        score += 0.12
    
    # 6. Too many interactive elements (cognitive overload)
    if interactive_count > 60:
        issues.append('too_many_interactive')
        score += 0.14
    elif interactive_count > 40:
        score += 0.05
    
    # 7. Cluttered UI (too many fixed elements)
    fixed_elements = layout.get('fixedElements', 0)
    if fixed_elements > 8:
        issues.append('cluttered_ui')
        score += 0.10
    elif fixed_elements > 5:
        score += 0.03
    
    # 8. Poor content density (mostly whitespace)
    if whitespace_ratio > 0.88 and word_count < 150:
        issues.append('poor_content_density')
        score += 0.11
    
    # 9. Insufficient ARIA labels
    focusable_count = len(focusable)
    if interactive_count > 15 and focusable_count < interactive_count * 0.5:
        issues.append('insufficient_aria')
        score += 0.13
    
    # 10. No semantic HTML (flex/grid containers > semantic elements)
    flex_containers = layout.get('flexContainers', 0)
    grid_containers = layout.get('gridContainers', 0)
    semantic_count = heading_count + len(dom_info.get('forms', []))
    
    if (flex_containers + grid_containers) > semantic_count and total_elements > 400:
        issues.append('no_semantic_html')
        score += 0.08
    
    # ========== SCORE NORMALIZATION ==========
    
    # Cap score at 0.95 (realistic range)
    score = min(score, 0.95)
    
    # Add small random noise for realism (±0.02)
    noise = random.uniform(-0.02, 0.05)
    final_score = min(1.0, max(0.0, score + noise))
    
    return {
        'issues': issues if issues else ['none'],
        'usability_score': round(final_score, 3)
    }


def process_captured_metadata():
    """
    Find all captured JSON metadata files and convert to labeled format.
    
    Expected structure:
        data/raw/session_YYYYMMDD_HHMMSS/
            ├── capture_id.jpg
            ├── capture_id.json  ← metadata file
            └── ...
    """
    
    print("=" * 70)
    print("🏷️  LABELING CAPTURED DOM DATA")
    print("=" * 70)
    
    # Find all metadata JSON files
    json_files = list(RAW_DIR.rglob("*.json"))
    
    # Filter out flow_graph.json
    json_files = [f for f in json_files if f.name != 'flow_graph.json']
    
    if not json_files:
        print(f"\n⚠️  No metadata JSON files found in {RAW_DIR}")
        print("   Run capture script first to generate data.")
        return
    
    print(f"\n📂 Found {len(json_files)} metadata files")
    print(f"   Location: {RAW_DIR}")
    
    output_rows = []
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Extract components
            capture_id = metadata.get('capture_id', 'unknown')
            dom_info = metadata.get('dom', {})
            visual_metrics = metadata.get('visual_metrics', {})
            
            # Skip if no DOM info
            if not dom_info or 'error' in dom_info:
                print(f"  ⚠️  Skipping {capture_id} - no valid DOM data")
                continue
            
            # Check for image file
            img_file = json_file.parent / f"{capture_id}.png"
            if not img_file.exists():
                img_file = json_file.parent / f"{capture_id}.jpg"
            
            if not img_file.exists():
                print(f"  ⚠️  No image for {capture_id}")
                continue
            
            # Analyze DOM
            label_info = analyze_dom_for_issues(dom_info, visual_metrics)
            
            # Format output row
            output_rows.append([
                str(img_file),
                str(json_file),
                label_info['usability_score'],
                ';'.join(label_info['issues'])
            ])
            
            print(f"  ✓ {capture_id}")
            print(f"    └─ Score: {label_info['usability_score']:.3f} | Issues: {', '.join(label_info['issues'])}")
        
        except Exception as e:
            print(f"  ❌ Error processing {json_file}: {e}")
            continue
    
    # Save to CSV
    if not output_rows:
        print("\n❌ No valid captures to label")
        return
    
    out_csv = OUT_DIR / "metadata.csv"
    
    try:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'dom_json', 'usability_score', 'issues'])
            writer.writerows(output_rows)
        
        print("\n" + "=" * 70)
        print("✅ LABELING COMPLETE")
        print("=" * 70)
        print(f"\n📊 Statistics:")
        print(f"   Total labeled: {len(output_rows)}")
        print(f"   Output: {out_csv}")
        
        # Score distribution
        scores = [float(r[2]) for r in output_rows]
        print(f"\n📈 Score Distribution:")
        print(f"   0.0-0.2 (Excellent):  {len([s for s in scores if 0.0 <= s <= 0.2])}")
        print(f"   0.2-0.4 (Good):       {len([s for s in scores if 0.2 < s <= 0.4])}")
        print(f"   0.4-0.6 (Fair):       {len([s for s in scores if 0.4 < s <= 0.6])}")
        print(f"   0.6-0.8 (Poor):       {len([s for s in scores if 0.6 < s <= 0.8])}")
        print(f"   0.8-1.0 (Very Poor):  {len([s for s in scores if 0.8 < s <= 1.0])}")
        
        # Issue frequency
        issue_counts = {}
        for row in output_rows:
            issues_str = row[3]
            if issues_str != 'none':
                for issue in issues_str.split(';'):
                    issue = issue.strip()
                    if issue:
                        issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        if issue_counts:
            print(f"\n🏷️  Top Issues Found:")
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                pct = 100 * count / len(output_rows)
                print(f"   {issue}: {count} ({pct:.1f}%)")
        
        print(f"\n📋 Sample Labeled Results (first 5):")
        for row in output_rows[:5]:
            issues = row[3][:50]
            print(f"   {row[2]:.2f} | {issues}")
        
        print(f"\n✨ Dataset ready for training!")
        print(f"   Next step: python models/train_tabular.py")
    
    except Exception as e:
        print(f"\n❌ Error writing CSV: {e}")
        return


if __name__ == "__main__":
    process_captured_metadata()