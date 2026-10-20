# Multi-stage production Dockerfile for ThermoFisher Scientific CoA Chat Agent
FROM python:3.12-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt pyproject.toml setup.py README.md ./
COPY thermofisher_agent ./thermofisher_agent
COPY app.py ./

RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM python:3.12-slim AS runtime

# Create non-root runtime user for hardened container security
RUN groupadd --gid 10001 coaagent && \
    useradd --uid 10001 --gid coaagent --shell /bin/bash --create-home coaagent

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    THERMOFISHER_MEMORY_DB=/tmp/thermofisher_session_memory.sqlite3

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY thermofisher_agent ./thermofisher_agent
COPY app.py ./

RUN chown -R coaagent:coaagent /app
USER coaagent

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
