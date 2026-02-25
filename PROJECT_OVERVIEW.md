# UXLens — ML-Powered UX Evaluation System

### BTP 2026 · Deep Learning for Web Usability

---

## 📌 What It Does

**UXLens** is a fully automated web application that audits any website's usability using machine learning.  
You paste a URL → it runs a full AI pipeline → you get a prioritised UX report in ~15 seconds.

**No manual uploads. No configuration. Fully automated.**

---

## 🧠 ML Pipeline

```
URL Input
   ↓
[1] Playwright captures full-page screenshot + DOM JSON
   ↓
[2] CNN model  →  Issue Probability  (is this page problematic?)
[3] XGBoost    →  Usability Score    (DOM structural quality)
   ↓
[4] Severity Index = prob_issue × (1 − usability_score)
   ↓
[5] Gemini 2.0 Flash → Prioritised UX Recommendations
   ↓
[6] HTML Results Page + Downloadable Markdown Report
```

### Models Used

| Model                    | Input                            | Output                  |
| ------------------------ | -------------------------------- | ----------------------- |
| CNN (`cnn_model.pth`)    | Full-page screenshot (PIL image) | `prob_issue` (0–1)      |
| XGBoost (`xgb_dom.json`) | DOM feature vector               | `usability_score` (0–1) |
| Gemini 2.0 Flash         | Metrics + DOM snippet            | JSON recommendations    |

### Severity Classification

| Severity Index | Label     |
| -------------- | --------- |
| > 2.0          | 🔴 HIGH   |
| 0.8 – 2.0      | 🟡 MEDIUM |
| < 0.8          | 🟢 LOW    |

---

## 📁 Project Structure

```
btp2026/
├── app/
│   ├── main.py                        ← FastAPI entry point
│   ├── services/
│   │   ├── capture_service.py         ← Playwright browser capture
│   │   ├── inference_service.py       ← CNN + XGBoost inference
│   │   ├── recommendation_service.py  ← Gemini 2.0 Flash LLM
│   │   └── report_service.py          ← Markdown report generator
│   ├── templates/
│   │   ├── index.html                 ← Home page
│   │   └── result.html                ← Results page
│   └── static/
│       ├── styles.css                 ← Dark-mode design system
│       └── screenshots/               ← Captured screenshots (auto)
├── models/
│   ├── cnn_model.pth                  ← Trained CNN weights
│   └── xgb_dom.json                   ← Trained XGBoost model
├── data/captures/                     ← DOM JSON snapshots (auto)
├── reports/                           ← Generated reports (auto)
├── .env                               ← API keys
└── PROJECT_OVERVIEW.md                ← This file
```

---

## 🔧 Tech Stack

| Layer              | Technology                     |
| ------------------ | ------------------------------ |
| Web Framework      | FastAPI + Uvicorn              |
| Browser Automation | Playwright (Chromium headless) |
| Deep Learning      | PyTorch (CNN)                  |
| Tabular ML         | XGBoost                        |
| LLM                | Google Gemini 2.0 Flash        |
| Frontend           | Jinja2 + Vanilla HTML/CSS      |
| Environment        | Python 3.10+                   |

---

## ▶️ How to Run

**Step 1 — Install dependencies (first time only)**

```bash
pip install fastapi uvicorn playwright torch torchvision xgboost
pip install google-generativeai python-dotenv jinja2 aiofiles opencv-python
playwright install chromium
```

**Step 2 — Set your Gemini API key**

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

_(Without a key, it falls back to generic recommendations automatically.)_

**Step 3 — Start the server**

```bash
cd D:\BTP\btp2026
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Step 4 — Open in browser**

```
http://localhost:8000
```

Paste any URL, click **Analyze →**, wait ~15 seconds, get your report.

---

## 📊 Sample Results

### `https://www.nsut.ac.in/en/home`

| Metric            | Value               |
| ----------------- | ------------------- |
| Issue Probability | **94.45%**          |
| Usability Score   | 0.3693              |
| Severity Index    | **2.557 — 🔴 HIGH** |

**Recommendations generated:**

1. 🔴 _Critical_ — Conduct user testing to identify friction points
2. 🔴 _Critical_ — Add descriptive alt text to all images and aria-labels
3. 🟡 _Important_ — Ensure a single prominent H1 heading
4. 🟡 _Important_ — Compress and lazy-load images; defer non-critical scripts

### `https://example.com`

| Metric            | Value                 |
| ----------------- | --------------------- |
| Issue Probability | 40.1%                 |
| Usability Score   | 0.3693                |
| Severity Index    | **1.085 — 🟡 MEDIUM** |

---

## 🎨 UI Features

- **Premium dark-mode** design (`#080d1a` base, glassmorphism cards)
- **Animated loading overlay** — shows pipeline progress while analyzing
- **Live page screenshot** — captured directly from the target site
- **Animated metric bars** — visual progress bars for each ML score
- **Colour-coded severity banner** — instant visual severity classification
- **Downloadable Markdown report** — shareable audit document

---

## ⚙️ API Endpoints

| Method | Route                  | Description                   |
| ------ | ---------------------- | ----------------------------- |
| `GET`  | `/`                    | Homepage                      |
| `POST` | `/analyze`             | Run full ML pipeline on a URL |
| `GET`  | `/report/{session_id}` | Download Markdown report      |

### POST `/analyze` — Form Data

```
url=https://example.com
Content-Type: application/x-www-form-urlencoded
```

### Response

Renders `result.html` with:

- `metrics` — `{prob_issue, usability_score, severity}`
- `recommendations` — list of `{priority, issue_type, fix, explanation}`
- `severity_class` — `"high"` | `"medium"` | `"low"`
- `screenshot_url` — relative path to captured PNG
- `session_id` — unique session for report download

---

## 🔑 Key Engineering Decisions

### 1. Sync Playwright in ThreadPoolExecutor

Playwright's async API raises `NotImplementedError` when nested inside FastAPI's uvicorn event loop on Windows. The fix: run Playwright's **sync API** in a `ThreadPoolExecutor` via `asyncio.run_in_executor()`.

```python
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(_executor, _do_capture, url, img_path, dom_path)
```

### 2. Model Loading at Startup

Both CNN and XGBoost models are loaded **once** at server startup using FastAPI's `lifespan` context manager — not on every request.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    inference_service.load_models()
    yield
```

### 3. Always-Populated Metrics Dict

Metrics are pre-initialised with zeros so Jinja2 templates never throw `UndefinedError` even if the pipeline fails mid-way.

```python
metrics = {"prob_issue": 0.0, "usability_score": 0.0, "severity": 0.0}
```

---

## 👥 Project Info

|                       |                                                   |
| --------------------- | ------------------------------------------------- |
| **Project**           | BTP 2026                                          |
| **System**            | ML UX Evaluation System                           |
| **Stack**             | FastAPI · PyTorch · XGBoost · Gemini · Playwright |
| **Server**            | `uvicorn app.main:app --port 8000`                |
| **Avg Response Time** | ~15 seconds per URL                               |

---

_Built with FastAPI · PyTorch · XGBoost · Google Gemini 2.0 Flash · Playwright Chromium_
