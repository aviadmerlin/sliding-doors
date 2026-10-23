FROM python:3.13-slim

WORKDIR /app

# curl only for the HEALTHCHECK below (useful for local `docker run` sanity
# checks -- Azure Container Apps uses its own HTTP probes, configured
# separately at deploy time, not this directive).
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code plus the precomputed data/model/assets (see README "Cold start
# and the fitted model") -- the app never refits anything at startup, it
# only reads these files. Never copies .env (see .dockerignore); Azure
# OpenAI credentials are injected as Container App secrets at runtime.
COPY app.py .
COPY src/ src/
COPY data/ data/
COPY assets/ assets/
COPY .streamlit/ .streamlit/

EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]
