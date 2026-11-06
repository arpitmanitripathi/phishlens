# PhishLens: production image. Only what the app needs: no PyTorch, no data, no notebooks.
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime (libgomp) on Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv installs the exact versions from uv.lock (same as on the laptop).
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1

WORKDIR /app

# 1. Dependencies first (cached by Docker until pyproject.toml / uv.lock change).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# 2. Then the code and the trained models.
COPY src ./src
COPY app ./app
COPY models ./models
RUN uv sync --frozen --no-dev

# Run as a normal user, not root (if the app were ever compromised, it could do less).
# The files stay owned by root: the app only needs to read them.
RUN useradd --create-home phishlens
USER phishlens

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Render sets $PORT; locally it defaults to 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
