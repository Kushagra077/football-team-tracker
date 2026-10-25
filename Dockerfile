# Football Team Tracker in a clean Linux container: locked dependencies, tests as the default command.
#   docker build -t football-team-tracker .
#   docker run --rm football-team-tracker
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# OpenCV's shared libraries are missing from the slim image
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first: this layer is rebuilt only when pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --extra dev --no-install-project

# Then the project itself (README.md is read by the package metadata).
COPY README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
RUN uv sync --locked --extra dev

CMD ["uv", "run", "pytest", "-m", "not integration", "-q"]
