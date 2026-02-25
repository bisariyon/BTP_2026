"""
app/services/report_service.py

Generates the downloadable Markdown report from analysis results.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime

REPORTS_DIR = pathlib.Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def generate_markdown_report(
    url: str,
    page_title: str,
    metrics: dict,
    recommendations: list[dict],
    session_id: str,
) -> tuple[str, str]:
    """
    Build a Markdown report and save it to disk.

    Returns:
        (report_path, markdown_text)
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    severity = metrics.get("severity", 0)
    if severity > 2.0:
        sev_label = "🔴 HIGH"
    elif severity > 0.8:
        sev_label = "🟡 MEDIUM"
    else:
        sev_label = "🟢 LOW"

    lines = [
        f"# UX Evaluation Report",
        f"",
        f"**URL:** {url}",
        f"**Page Title:** {page_title}",
        f"**Generated:** {now}",
        f"**Session ID:** {session_id}",
        f"",
        f"---",
        f"",
        f"## 📊 ML Metrics",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Issue Probability | `{metrics.get('prob_issue', 0):.2%}` |",
        f"| Usability Score | `{metrics.get('usability_score', 0):.4f}` |",
        f"| Severity Index | `{metrics.get('severity', 0):.4f}` — **{sev_label}** |",
        f"",
        f"---",
        f"",
        f"## 🛠 Prioritised Recommendations",
        f"",
    ]

    for i, rec in enumerate(recommendations, 1):
        priority = rec.get("priority", 2)
        badge = {1: "🔴 Critical", 2: "🟡 Important", 3: "🟢 Minor"}.get(priority, "⚪")
        lines += [
            f"### {i}. {rec.get('issue_type', 'Recommendation')} — {badge}",
            f"",
            f"**Fix:** {rec.get('fix', '')}",
            f"",
            f"**Why it matters:** {rec.get('explanation', '')}",
            f"",
        ]

    lines += [
        f"---",
        f"*Report generated automatically by the ML UX Evaluation System.*",
    ]

    md_text = "\n".join(lines)

    report_path = REPORTS_DIR / f"{session_id}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    return str(report_path), md_text
