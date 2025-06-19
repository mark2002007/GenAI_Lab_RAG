FROM python:3.11-slim

# (system deps for Chromium if your ingest step needs it; optional once ingest is baked)
RUN apt-get update && \
    apt-get install -y --no-install-recommends chromium-driver && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy application code
COPY src/ ./src/

# 3) Copy prebuilt data into the container
COPY output/ ./output/

# 4) Expose the Streamlit port
EXPOSE 8501

# 5) Default command to run your app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.headless=true"]