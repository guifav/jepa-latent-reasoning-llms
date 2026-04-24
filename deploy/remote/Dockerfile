FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates tmux rsync && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/jepa

COPY requirements-remote.txt /workspace/jepa/requirements-remote.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /workspace/jepa/requirements-remote.txt

COPY src /workspace/jepa/src
COPY scripts /workspace/jepa/scripts
COPY configs /workspace/jepa/configs
COPY data /workspace/jepa/data
COPY analysis /workspace/jepa/analysis
COPY requirements-phase1.txt /workspace/jepa/requirements-phase1.txt

ENTRYPOINT ["python", "/workspace/jepa/scripts/phase1_train.py"]
