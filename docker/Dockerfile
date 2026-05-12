# VLSI Circuit Partitioning Framework
# Provides a reproducible environment with Python + all dependencies.
# OpenROAD is NOT bundled here; mount it or install via your org's image.

FROM python:3.11-slim

LABEL maintainer="vlsi-partition-framework"
LABEL description="Metaheuristic VLSI circuit partitioning (ISCAS/ISPD98 benchmarks)"

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        git \
        bash \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
WORKDIR /app
COPY . /app

# Pre-create output directories
RUN mkdir -p benchmarks/circuits results/solutions docs/figures

# Make scripts executable
RUN chmod +x scripts/fetch_benchmarks.sh scripts/run_all.sh

# ---------------------------------------------------------------------------
# Default command
# ---------------------------------------------------------------------------
# Override with docker run ... python src/main.py --help
CMD ["bash"]
