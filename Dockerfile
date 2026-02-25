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
    wget -q -O models/cnn_model.pth "https://drive.google.com/file/d/1kO2I1PgcScKPEl6aY5Y5h3XUoWQt0rIB/view?usp=drive_link" && \
    wget -q -O models/xgb_dom.json  "https://drive.google.com/file/d/1U_-lrQzIshSGexePA2-foqFTQR3Cd_HQ/view?usp=drive_link"

# Create runtime directories
RUN mkdir -p app/static/screenshots data/captures reports

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]