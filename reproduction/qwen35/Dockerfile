# Build from the REPOSITORY ROOT, not this directory.
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
FROM ${CUDA_IMAGE}
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev build-essential git ca-certificates libnuma1 \
    && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    HF_HOME=/data/hf \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false
RUN uv venv --python /usr/bin/python3 /opt/venv
COPY reproduction/qwen35/requirements.in reproduction/qwen35/requirements.lock /opt/
# No GPU is required at build time. PyPI's pinned torch uses CUDA 13.0.
# Install the checked-in, hash-locked Linux/Python 3.12 resolution. Pin image
# digests after the first successful build as described in the guide.
RUN uv pip sync --python /opt/venv/bin/python /opt/requirements.lock \
    && uv pip check --python /opt/venv/bin/python
WORKDIR /workspace
COPY reproduction/qwen35/reproduce.py /opt/reproduce.py
ENTRYPOINT ["/opt/venv/bin/python", "/opt/reproduce.py"]
CMD ["--help"]
