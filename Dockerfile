FROM node:20-bookworm-slim AS frontend-build

WORKDIR /src/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BRAINOS_HOST=0.0.0.0 \
    BRAINOS_PORT=8765 \
    HF_HOME=/data/huggingface \
    BRAINOS_REPLAY_DIR=/data/replay_sessions

WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
COPY backend/requirements-runtime.txt /app/backend/requirements-runtime.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" \
    && pip install --no-cache-dir -r /app/backend/requirements-runtime.txt
COPY backend/ /app/backend/
COPY alembic.ini /app/alembic.ini
COPY --from=frontend-build /src/frontend/dist /app/frontend/dist

RUN mkdir -p /data/huggingface /data/replay_sessions /data/brainos
VOLUME ["/data/huggingface", "/data/replay_sessions", "/data/brainos"]
EXPOSE 8765

WORKDIR /app/backend
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/ready', timeout=5)"
CMD ["python", "run.py"]
