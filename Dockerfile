# syntax=docker/dockerfile:1
# Local inference only: model assets are mounted read-only at runtime.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copy pinned dependencies separately for Docker layer caching.
COPY requirements/runtime.txt requirements/api.txt /app/requirements/
RUN python -m pip install --no-cache-dir \
        -r /app/requirements/runtime.txt \
        -r /app/requirements/api.txt \
    && python -m pip check \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser

# Never COPY . or baked-in datasets, fitted models, or secrets.
COPY app/ /app/app/
COPY src/ /app/src/
COPY config/inference.json /app/config/inference.json

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
