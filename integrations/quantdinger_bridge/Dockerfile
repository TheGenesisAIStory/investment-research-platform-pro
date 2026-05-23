FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/ml4t \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/ml4t

COPY requirements-quantdinger-bridge.txt /tmp/requirements-quantdinger-bridge.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements-quantdinger-bridge.txt \
    && useradd --create-home --shell /usr/sbin/nologin appuser

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["uvicorn", "integrations.quantdinger_bridge.ml_service:app", "--host", "0.0.0.0", "--port", "8000"]
