FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /workspace

ENV UV_LINK_MODE=copy
ENV PATH="/workspace/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-cache

COPY . .

CMD ["/bin/bash"]