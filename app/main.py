"""
app/main.py — FastAPI entry-point for the ML UX Evaluation System.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Load .env before importing services
load_dotenv()

from app.services import inference_service
from app.services.capture_service import capture_single_url
from app.services.recommendation_service import generate_recommendations
from app.services.report_service import generate_markdown_report

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(name)s │ %(message)s")
logger = logging.getLogger(__name__)

# ─── Lifespan: load models once at startup ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML models …")
    try:
        inference_service.load_models()
        logger.info("✅ Models ready.")
    except Exception as exc:
        logger.error("⚠️  Model loading failed: %s", exc)
    yield  # server runs
    logger.info("Server shutting down.")

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="ML UX Evaluation System", version="1.0.0", lifespan=lifespan)

# Static files (screenshots, CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="app/templates")


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, url: str = Form(...)):
    """
    Full pipeline:
      1. Capture screenshot + DOM
      2. Run CNN + XGBoost inference
      3. Call Gemini for recommendations
      4. Generate Markdown report
      5. Render result page
    """
    # Normalise URL
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    error_msg = None
    capture_result = None
    metrics = {"prob_issue": 0.0, "usability_score": 0.0, "severity": 0.0}
    recommendations = []
    report_md = ""
    session_id = "unknown"
    page_title = ""
    dom_snippet = {}

    try:
        # ── 1. Capture ────────────────────────────────────────────────────────
        logger.info("Capturing URL: %s", url)
        capture_result = await capture_single_url(url)
        session_id = capture_result["session_id"]
        page_title = capture_result.get("page_title", "")

        # Load DOM for context
        with open(capture_result["dom_json_path"], "r", encoding="utf-8") as f:
            dom_snippet = json.load(f)

        # ── 2. Inference ──────────────────────────────────────────────────────
        logger.info("Running inference …")
        metrics = inference_service.run_inference(
            capture_result["image_path"],
            capture_result["dom_json_path"],
        )
        metrics["url"] = url

        # ── 3. Gemini recommendations ─────────────────────────────────────────
        logger.info("Calling Gemini …")
        recommendations = generate_recommendations(metrics, dom_snippet)

        # ── 4. Markdown report ────────────────────────────────────────────────
        report_path, report_md = generate_markdown_report(
            url=url,
            page_title=page_title,
            metrics=metrics,
            recommendations=recommendations,
            session_id=session_id,
        )
        logger.info("Report saved to %s", report_path)

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        error_msg = str(exc)

    # Severity classification
    severity_val = metrics.get("severity", 0)
    if severity_val > 2.0:
        severity_class = "high"
        severity_label = "HIGH"
    elif severity_val > 0.8:
        severity_class = "medium"
        severity_label = "MEDIUM"
    else:
        severity_class = "low"
        severity_label = "LOW"

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "url": url,
            "page_title": page_title,
            "session_id": session_id,
            "screenshot_url": capture_result["screenshot_url"] if capture_result else None,
            "metrics": metrics,
            "recommendations": recommendations,
            "severity_class": severity_class,
            "severity_label": severity_label,
            "report_md": report_md,
            "error": error_msg,
            "dom": dom_snippet,
        },
    )


@app.get("/report/{session_id}", response_class=Response)
async def download_report(session_id: str):
    """Download the Markdown report for a session."""
    report_path = Path("reports") / f"{session_id}_report.md"
    if not report_path.exists():
        return Response(content="Report not found.", status_code=404, media_type="text/plain")

    content = report_path.read_text(encoding="utf-8")
    return Response(
        content=content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{session_id}_report.md"'},
    )
