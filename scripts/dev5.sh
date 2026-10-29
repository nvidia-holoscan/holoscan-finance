#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run one command in the Holoscan 5.0 image against this checkout:
#   scripts/dev5.sh 'python3 -m pytest -q -m "not slow" tests/python'
# Non-root with private IPC (never --ipc=host), off the DAQIRI lcores 3 and 9, and on the
# RTX PRO 6000 (PCI bus order). The timeout runs inside the container because a host-side
# timeout does not stop bash as PID 1 (RUN5_TIMEOUT, default 1500 s; exit 137). Test
# dependencies and holoscan-cli (the image has no `holoscan` command), at the versions
# requirements-dev.txt pins (as CI installs them), go into the gitignored .cache/pyuser5 under
# flock, so parallel runs share one user base. CPU-only: dev5cpu.sh.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN5_PIP='python3 -m pip install --user -q --disable-pip-version-check -r requirements-cli.txt -c requirements-dev.txt scipy threadpoolctl fastapi uvicorn anyio httpx pytest-timeout ruff numba >/dev/null'   # ONE line: it runs via bash -c
exec docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=1 -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp \
  -e PYTHONUSERBASE="$REPO/.cache/pyuser5" -e CUPY_CACHE_DIR="$REPO/.cache/cupy" \
  -e RUN5_CMD="$*" -e RUN5_TIMEOUT="${RUN5_TIMEOUT:-1500}" -e RUN5_PIP="$RUN5_PIP" \
  --shm-size=2g --ulimit stack=33554432 --cpuset-cpus=0-2,4-8,10-13 -u "$(id -u):$(id -g)" \
  -v "$REPO:$REPO" -w "$REPO" --entrypoint bash nvcr.io/nvidia/clara-holoscan/holoscan:v5.0.0-cuda13 -c '
    export PATH=$PYTHONUSERBASE/bin:$PATH PYTHONPATH=$PYTHONPATH:'"$REPO"'/python:'"$REPO"'/tests/python
    mkdir -p $PYTHONUSERBASE && flock $PYTHONUSERBASE.lock bash -c "$RUN5_PIP"   # parallel agents share the user base
    timeout -s KILL "$RUN5_TIMEOUT" bash -c "$RUN5_CMD"'
