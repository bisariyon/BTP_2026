import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd
import google.generativeai as genai
from models.inference import predict

# Configure Gemini client (reads GEMINI_API_KEY from env)
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyANFBnyP2err66sd9JSVR-dmg2hq8YjeLk"))

# Initialize model once
model = genai.GenerativeModel("gemini-2.0-flash")  # or gemini-1.5-pro if you want higher quality

def make_prompt(issue_data):
    return f"""
You are a UX expert. Analyze the following website metrics and recommend fixes:

{json.dumps(issue_data, indent=2)}

Write a prioritized list of recommendations for designers and developers.
Each recommendation should have:
- issue_type
- priority (1=high, 3=low)
- fix (actionable suggestion)
- explanation (why this matters)
"""

def generate_recommendations(image_path, dom_json, issues=""):
    result = predict(image_path, dom_json)

    # Convert issues safely to list
    if isinstance(issues, float) or issues is None or pd.isna(issues):
        issue_list = []
    else:
        issue_list = str(issues).split(";")

    issue_data = {
        "image": image_path,
        "prob_issue": float(result["prob_issue"]),
        "usability_score": float(result["usability_score"]),
        "severity": float(result["severity"]),
        "detected_issues": issue_list
    }

    prompt = make_prompt(issue_data)

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Failed to generate recommendations."
    
import pathlib

REPORTS_DIR = pathlib.Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    meta = pd.read_csv("data/labelled/metadata.csv").head(3)
    for _, row in meta.iterrows():
        recs = generate_recommendations(row["image_path"], row["dom_json"], row["issues"])
        
        # Print to terminal
        print("\n=== Recommendations for", row["image_path"], "===\n")
        print(recs)

        # Save to Markdown file
        filename = pathlib.Path(row["image_path"]).stem + "_recommend.md"
        report_path = REPORTS_DIR / filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Recommendations for {row['image_path']}\n\n")
            f.write(recs)

        print(f"✅ Saved report to {report_path}")
