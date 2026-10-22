#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# dev5.sh for the CPU-only stub tier: the same image, flags and user base, but no GPU runtime
# and no NVIDIA/CUDA variables. A stub libcuda.so.1 lets `import holoscan` load; the script
# refuses (exit 3) if a real driver is visible, so a CPU-only result is never a GPU result.
#   scripts/dev5cpu.sh 'python3 -m pytest -q -m "sdk and not gpu and not slow" tests/python'
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN5_PIP='python3 -m pip install --user -q --disable-pip-version-check -r requirements-cli.txt -c requirements-dev.txt scipy threadpoolctl fastapi uvicorn anyio httpx duckdb pyarrow pytest-timeout ruff numba >/dev/null'   # ONE line, as in dev5.sh
exec docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp \
  -e PYTHONUSERBASE="$REPO/.cache/pyuser5" -e CUPY_CACHE_DIR=/tmp/cupy \
  -e RUN5_CMD="$*" -e RUN5_TIMEOUT="${RUN5_TIMEOUT:-1500}" -e RUN5_PIP="$RUN5_PIP" \
  --shm-size=2g --ulimit stack=33554432 --cpuset-cpus=0-2,4-8,10-13 -u "$(id -u):$(id -g)" \
  -v "$REPO:$REPO" -w "$REPO" --entrypoint bash nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13 -c '
    export PATH=$PYTHONUSERBASE/bin:$PATH PYTHONPATH=$PYTHONPATH:'"$REPO"'/python:'"$REPO"'/tests/python
    if python3 -c "import ctypes; ctypes.CDLL(\"libcuda.so.1\")" 2>/dev/null; then
      echo "dev5cpu.sh: a real CUDA driver is visible; refusing" >&2; exit 3; fi
    mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
    export LD_LIBRARY_PATH=/tmp/stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    mkdir -p $PYTHONUSERBASE && flock $PYTHONUSERBASE.lock bash -c "$RUN5_PIP"
    timeout -s KILL "$RUN5_TIMEOUT" bash -c "$RUN5_CMD"'
