# UXLens — Deployment Guide

> **Why not Vercel?**  
> Playwright needs a real browser installed, PyTorch weighs ~1 GB, and our pipeline takes ~15s —
> all of which exceed Vercel's serverless limits. Use **Railway** instead (free tier available).

---

## Pre-Requisites (do these once, before anything else)

- [ ] GitHub account
- [ ] Railway account → [railway.app](https://railway.app) (sign in with GitHub)
- [ ] Your model files hosted somewhere (see Step 1 below)

---

## Step 1 — Host Your Model Files

Your `.gitignore` excludes the large model binaries. You need to make them downloadable at deploy time.

### Option A — Google Drive (simplest)

1. Upload `models/cnn_model.pth` and `models/xgb_dom.json` to Google Drive
2. Right-click each → **Share** → "Anyone with the link" → **Viewer**
3. Copy the shareable link — you'll use it in the startup script below

### Option B — HuggingFace Hub (recommended for ML projects)

1. Go to [huggingface.co](https://huggingface.co) → New Model → name it `uxlens-models`
2. Upload both files via the web UI
3. Your download URLs will be:
   ```
   https://huggingface.co/YOUR_USERNAME/uxlens-models/resolve/main/cnn_model.pth
   https://huggingface.co/YOUR_USERNAME/uxlens-models/resolve/main/xgb_dom.json
   ```

---

## Step 2 — Add a `Dockerfile`

Create this file in the root of your project (`d:\BTP\btp2026\Dockerfile`):

```dockerfile
FROM python:3.11-slim

# Install OS dependencies required by Playwright/Chromium
RUN apt-get update && apt-get install -y \
    wget curl \
    libglib2.0-0 libnss3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libasound2 libpango-1.0-0 \
    libcairo2 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser
RUN playwright install chromium --with-deps

# Copy app source (gitignored files like models/ won't be here)
COPY . .

# Download model files at build time
# Replace the URLs below with your actual Google Drive or HuggingFace URLs
RUN mkdir -p models && \
    wget -q -O models/cnn_model.pth "YOUR_CNN_MODEL_DOWNLOAD_URL" && \
    wget -q -O models/xgb_dom.json  "YOUR_XGB_MODEL_DOWNLOAD_URL"

# Create runtime directories
RUN mkdir -p app/static/screenshots data/captures reports

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> ⚠️ Replace `YOUR_CNN_MODEL_DOWNLOAD_URL` and `YOUR_XGB_MODEL_DOWNLOAD_URL`
> with the direct download links from Step 1.

---

## Step 3 — Create `requirements.txt`

Run this in your terminal to generate it automatically:

```bash
cd D:\BTP\btp2026
pip freeze > requirements.txt
```

Or create it manually with the key packages:

```
fastapi
uvicorn[standard]
jinja2
python-multipart
playwright
torch
torchvision
xgboost
numpy
Pillow
opencv-python-headless
groq
python-dotenv
aiofiles
```

> Use `opencv-python-headless` (not `opencv-python`) — the headless version works on
> servers without a display.

---

## Step 4 — Push to GitHub

```bash
cd D:\BTP\btp2026

# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit — UXLens ML UX Evaluation System"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/uxlens.git
git branch -M main
git push -u origin main
```

> ✅ The `.gitignore` will automatically exclude `.env`, model files, screenshots,
> raw data, and reports — only source code gets pushed.

---

## Step 5 — Deploy on Railway

1. Go to [railway.app](https://railway.app) → **New Project**
2. Click **Deploy from GitHub repo**
3. Select your `uxlens` repository
4. Railway auto-detects the `Dockerfile` and starts building

### Add Environment Variables

In Railway dashboard → your project → **Variables** tab → add:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Set the Port

Railway reads the `PORT` env var automatically. Your server already binds to `0.0.0.0:8000`
so it will work. If Railway assigns a different port, update:

```
PORT=8000
```

---

## Step 6 — Get Your Live URL

Once deployed, Railway gives you a public URL like:

```
https://uxlens-production.up.railway.app
```

Share this URL — anyone can paste a URL and get a UX audit instantly.

---

## Troubleshooting

| Problem                            | Fix                                                                                          |
| ---------------------------------- | -------------------------------------------------------------------------------------------- |
| Build fails — Playwright not found | Make sure `playwright install chromium --with-deps` is in the Dockerfile                     |
| Model download fails               | Verify the download URL works in a browser first (must be a **direct** link, not a redirect) |
| `500` error on `/analyze`          | Check Railway logs → usually a missing env var or model file                                 |
| Timeout on analysis                | Railway has no timeout limit on free tier — should be fine at ~15s                           |
| `ModuleNotFoundError`              | Check `requirements.txt` has all packages                                                    |

---

## Estimated Build Time

| Step                 | Time                            |
| -------------------- | ------------------------------- |
| Docker image build   | ~5–8 min (first time)           |
| Playwright install   | ~2 min                          |
| Model download       | ~1–3 min (depends on file size) |
| **Total cold start** | **~10 min first deploy**        |

Subsequent deploys (after code changes) are **much faster** as Docker layers are cached.

---

_UXLens · BTP 2026 · FastAPI + PyTorch + XGBoost + Groq llama-3.1-8b-instant_
