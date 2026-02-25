"""
app/services/recommendation_service.py

Generates structured UX recommendations using Groq (llama-3.1-8b-instant).
"""

from __future__ import annotations

import json
import logging
import os
import re

from dotenv import load_dotenv
from groq import Groq

# ── commented out Gemini ─────────────────────────────────────────────────────
# import google.generativeai as genai
# if _api_key:
#     genai.configure(api_key=_api_key)
#     _gemini = genai.GenerativeModel("gemini-2.0-flash")
# else:
#     _gemini = None
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Initialise Groq client once ─────────────────────────────────────────────
_api_key = os.getenv("GROQ_API_KEY", "")
if _api_key:
    _groq = Groq(api_key=_api_key)
    logger.info("Groq client ready (llama-3.1-8b-instant).")
else:
    _groq = None
    logger.warning("GROQ_API_KEY not set — recommendations will use fallback.")


# ─── Prompt builder ──────────────────────────────────────────────────────────

def _build_prompt(metrics: dict, dom_snippet: dict) -> str:
    data = {
        "url": metrics.get("url", "unknown"),
        "prob_issue": metrics.get("prob_issue"),
        "usability_score": metrics.get("usability_score"),
        "severity_index": metrics.get("severity"),
        "page_title": dom_snippet.get("title", ""),
        "interactive_count": dom_snippet.get("interactive_count", 0),
        "images_count": dom_snippet.get("images_count", 0),
        "imagesWithoutAlt": dom_snippet.get("imagesWithoutAlt", 0),
        "linksWithoutText": dom_snippet.get("accessibility", {}).get("linksWithoutText", 0),
        "wordCount": dom_snippet.get("textDensity", {}).get("wordCount", 0),
        "totalElements": dom_snippet.get("layout", {}).get("totalElements", 0),
        "headings": [h.get("text", "") for h in dom_snippet.get("headings", [])[:5]],
    }

    return f"""You are a senior UX/accessibility expert reviewing an automated audit of a website.

Below are the ML-generated metrics and DOM statistics for the analysed page:

{json.dumps(data, indent=2)}

Scoring note:
- prob_issue: probability (0–1) that the page has usability problems (higher = worse).
- usability_score: XGBoost predicted usability (higher = better).
- severity_index: combined severity (higher = more urgent attention needed).

Return a JSON array (ONLY the JSON array, no markdown fences, no extra text) of 4–6 actionable recommendations.
Each element must have these exact keys:
  "issue_type"   : short category (e.g. "Accessibility", "Visual Hierarchy", "Performance")
  "priority"     : integer 1 (critical), 2 (important), or 3 (minor)
  "fix"          : one concrete, actionable change the developer/designer should make
  "explanation"  : why this matters for usability or accessibility

Order by priority ascending (most critical first)."""


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_recommendations(metrics: dict, dom_snippet: dict) -> list[dict]:
    """
    Calls Groq llama-3.1-8b-instant for page-specific UX recommendations.
    Falls back to generic recommendations if the API is unavailable.
    """
    if _groq is None:
        logger.warning("Groq unavailable — using fallback recommendations.")
        return _fallback_recommendations(metrics)

    prompt = _build_prompt(metrics, dom_snippet)

    try:
        response = _groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a UX/accessibility expert. "
                        "Always respond with a valid JSON array only — no markdown, no prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        text = response.choices[0].message.content.strip()

        # Strip any accidental markdown fences
        text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n?```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        parsed = json.loads(text)

        if isinstance(parsed, list):
            logger.info("Groq returned %d recommendations.", len(parsed))
            return parsed

        # Handle rare case where Groq wraps the array in an object
        for val in parsed.values():
            if isinstance(val, list):
                return val

        return _fallback_recommendations(metrics)

    except Exception as exc:
        logger.error("Groq call failed: %s", exc)
        return _fallback_recommendations(metrics)


# ─── Fallback ─────────────────────────────────────────────────────────────────

def _fallback_recommendations(metrics: dict) -> list[dict]:
    """Generic recommendations used when Groq is not available."""
    recs = []
    if metrics.get("prob_issue", 0) > 0.6:
        recs.append({
            "issue_type": "Usability Risk",
            "priority": 1,
            "fix": "Conduct user testing to identify friction points on the page.",
            "explanation": "The ML model flagged a high probability of usability issues.",
        })
    recs += [
        {
            "issue_type": "Accessibility",
            "priority": 1,
            "fix": "Add descriptive alt text to all images and aria-labels to icon-only links.",
            "explanation": "Missing alt text breaks screen reader compatibility and harms SEO.",
        },
        {
            "issue_type": "Visual Hierarchy",
            "priority": 2,
            "fix": "Ensure a single, prominent H1 heading and clear content sections.",
            "explanation": "Strong hierarchy guides users to the most important content first.",
        },
        {
            "issue_type": "Performance",
            "priority": 2,
            "fix": "Compress and lazy-load images; defer non-critical scripts.",
            "explanation": "Faster load times directly improve user retention and SEO ranking.",
        },
    ]
    return recs
