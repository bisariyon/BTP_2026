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

# Prevent memory fragmentation on low-RAM instances
ENV MALLOC_ARENA_MAX=2

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser
RUN playwright install chromium --with-deps

# Copy app source (gitignored files like models/ won't be here)
COPY . .

# Download model files at build time (direct Google Drive download links)
RUN mkdir -p models && \
    wget --no-check-certificate -q \
    "https://drive.usercontent.google.com/download?id=1kO2I1PgcScKPEl6aY5Y5h3XUoWQt0rIB&export=download&confirm=t" \
    -O models/cnn_model.pth && \
    wget --no-check-certificate -q \
    "https://drive.usercontent.google.com/download?id=1U_-lrQzIshSGexePA2-foqFTQR3Cd_HQ&export=download&confirm=t" \
    -O models/xgb_dom.json

# Create runtime directories
RUN mkdir -p app/static/screenshots data/captures reports

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]