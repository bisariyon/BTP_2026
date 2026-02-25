# UXLens — Deployment Guide (Render.com)

> **Platform:** [Render.com](https://render.com) — Docker-based, free tier, no credit card needed.  
> **Why not Vercel?** Playwright needs a real browser, PyTorch is ~1 GB, and our pipeline
> takes ~15 s — all beyond Vercel's serverless limits.

---

## Overview

```
Local machine  →  GitHub  →  Render.com  →  Live URL
```

Render reads your `Dockerfile`, builds a container, and serves it at a public HTTPS URL.

---

## Files Already in Your Project

| File               | Purpose                                   |
| ------------------ | ----------------------------------------- |
| `Dockerfile`       | Tells Render how to build and run the app |
| `requirements.txt` | Python dependencies                       |
| `render.yaml`      | Render service config                     |
| `.gitignore`       | Keeps secrets/models/data out of git      |

---

## Step 1 — Make Model Files Publicly Downloadable

Your model weights are gitignored (too large for git). The `Dockerfile` downloads them
from Google Drive at build time.

**Make them public:**

1. Open [drive.google.com](https://drive.google.com)
2. Right-click `cnn_model.pth` → **Share** → **"Anyone with the link"** → **Viewer** → Done
3. Repeat for `xgb_dom.json`

> ✅ Already done if you added the file IDs to the Dockerfile.  
> The Dockerfile uses direct download links (not viewer links) — format:  
> `https://drive.usercontent.google.com/download?id=FILE_ID&export=download&confirm=t`

---

## Step 2 — Push to GitHub

```bash
cd D:\BTP\btp2026

# First time setup
git init
git add .
git commit -m "Initial commit — UXLens ML UX Evaluation System"

# Create a new repo on github.com, then link it:
git remote add origin https://github.com/YOUR_USERNAME/uxlens.git
git branch -M main
git push -u origin main
```

> If you already have a repo and just made changes:
>
> ```bash
> git add .
> git commit -m "Add Render deployment config"
> git push
> ```

---

## Step 3 — Create Account on Render

1. Go to **[render.com](https://render.com)**
2. Click **Get Started for Free**
3. Sign up with your **GitHub account** (easiest — no card needed)

---

## Step 4 — Create a New Web Service

1. In the Render dashboard → click **New +** → **Web Service**
2. Click **Connect a repository** → select your `uxlens` GitHub repo
3. Render will auto-detect the `Dockerfile`

Fill in the settings:

| Setting           | Value                        |
| ----------------- | ---------------------------- |
| **Name**          | `uxlens` (or anything)       |
| **Region**        | Singapore (closest to India) |
| **Branch**        | `main`                       |
| **Runtime**       | Docker _(auto-detected)_     |
| **Instance Type** | **Free**                     |

---

## Step 5 — Add Environment Variables

Scroll down to **Environment Variables** → click **Add Environment Variable**:

| Key            | Value                                                               |
| -------------- | ------------------------------------------------------------------- |
| `GROQ_API_KEY` | your Groq API key from [console.groq.com](https://console.groq.com) |

> ⚠️ Never hardcode secrets — always add them here, never in your code.

---

## Step 6 — Deploy

Click **Create Web Service** → Render starts the build.

**What happens during the build (~10–15 min first time):**

```
[1] Pull python:3.11-slim base image
[2] Install OS packages (Playwright dependencies)
[3] pip install -r requirements.txt
[4] playwright install chromium --with-deps
[5] COPY source code
[6] wget model files from Google Drive
[7] Start uvicorn server
```

You can watch live logs in the Render dashboard.

---

## Step 7 — Get Your Live URL

Once the build succeeds, Render gives you a URL:

```
https://uxlens.onrender.com
```

Share this link — anyone can paste a URL and get an instant ML UX audit. ✅

---

## Updating the App Later

Every time you push to GitHub, Render **auto-redeploys**:

```bash
# Make your code changes, then:
git add .
git commit -m "describe your change"
git push
```

Render detects the push and rebuilds automatically.

---

## Important Notes

### ⚠️ Free Tier Limitation — Cold Starts

On the free tier, Render **sleeps the server after 15 minutes of inactivity**.
The first request after sleep takes **~30–60 seconds** to wake up.
Subsequent requests are instant.

To avoid this → upgrade to **Starter plan ($7/month)** which keeps the server always on.

### ⚠️ Model File Downloads

If `wget` fails to download the model files (e.g. Google Drive quota exceeded), the build
will fail. Alternative: host models on **HuggingFace Hub** (free, no quota issues):

```
https://huggingface.co/YOUR_USERNAME/uxlens-models/resolve/main/cnn_model.pth
```

### ✅ Logs

View real-time logs in Render dashboard → your service → **Logs** tab.  
Useful for debugging `500` errors after deploy.

---

## Troubleshooting

| Error                                 | Fix                                                          |
| ------------------------------------- | ------------------------------------------------------------ |
| Build fails at `wget` step            | Make Google Drive files public (Step 1)                      |
| `ModuleNotFoundError`                 | Check `requirements.txt` has all packages                    |
| `500` on `/analyze`                   | Check Render logs — likely missing `GROQ_API_KEY` env var    |
| Slow first response (~30s)            | Normal — free tier cold start                                |
| Playwright `BrowserType.launch` error | The Dockerfile installs all Chromium deps — check build logs |

---

## Quick Reference

```bash
# Push latest changes to trigger redeploy
git add . && git commit -m "update" && git push

# Local dev (unchanged)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

_UXLens · BTP 2026 · FastAPI + PyTorch + XGBoost + Groq llama-3.1-8b-instant · Deployed on Render_
