# Football Team Tracker in a clean Linux container: locked dependencies, tests as the default command.
#   docker build -t football-team-tracker .
#   docker run --rm football-team-tracker                      # the tests
#   docker run --rm -v "$PWD/clips:/clips" -v "$PWD/out:/app/reports" football-team-tracker \
#       python scripts/process_video.py /clips/my_clip.mp4 --device cpu   # track a clip
#
# CPU only. The lockfile pins the GPU build of PyTorch plus ~4 GB of NVIDIA libraries; a container
# without a GPU never uses them, so they are left out and the CPU build of the SAME torch version
# is installed from PyTorch's own index.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# OpenCV's shared libraries are missing from the slim image
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    VIRTUAL_ENV=/app/.venv PATH=/app/.venv/bin:$PATH

# Dependencies first: this layer is rebuilt only when pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv venv \
    && uv export --locked --extra dev --no-emit-project --no-hashes -o /tmp/locked.txt \
    && grep -viE '^(nvidia-|cuda-|triton|torch==|torchvision==)' /tmp/locked.txt > /tmp/cpu.txt \
    && TORCH=$(grep -iE '^torch==' /tmp/locked.txt | cut -d' ' -f1) \
    && VISION=$(grep -iE '^torchvision==' /tmp/locked.txt | cut -d' ' -f1) \
    && uv pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "$TORCH" "$VISION" \
    && uv pip install -r /tmp/cpu.txt
# torch goes in first: ultralytics requires it, and with it already installed uv does not fetch the GPU build.

# Then the project itself (README.md is read by the package metadata).
COPY README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
RUN uv pip install --no-deps -e .

# The detector weights, from the public detector repo, pinned to the commit that added them. They go
# where configs/paths.yaml expects them (next to the repo), so scripts/process_video.py runs as is.
ADD https://github.com/Kushagra077/football-detect-serve/raw/71824d8e00bfa1256568f333cf56ab4192abe766/models/football_detection_v3.pt \
    /football-detect-serve/models/football_detection_v3.pt

# Plain pytest, not `uv run`: uv run would re-sync the venv from the lockfile and bring the GPU libraries back.
CMD ["pytest", "-m", "not integration", "-q"]
