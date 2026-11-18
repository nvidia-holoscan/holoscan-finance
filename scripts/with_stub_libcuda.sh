#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run a command in the Holoscan 5.0 image on a machine with no GPU driver (the CI stub tier):
#   scripts/with_stub_libcuda.sh python3 -m pytest -q -m "sdk and not gpu and not slow"
# `import holoscan` links libcuda.so.1, so the CUDA toolkit's stub stands in for it; graphs
# compile and CPU operators run, and any real CUDA call fails. If a real driver loads, the
# script refuses (exit 3) so that a CPU-only result is never mistaken for a GPU one.
# scripts/dev5cpu.sh does the same for local runs.
set -euo pipefail
[[ $# -gt 0 ]] || { echo "usage: $0 COMMAND [ARG...]" >&2; exit 2; }
if python3 -c 'import ctypes; ctypes.CDLL("libcuda.so.1")' 2>/dev/null; then
    echo "with_stub_libcuda.sh: a real CUDA driver is visible; refusing" >&2
    exit 3
fi
STUB_LIBCUDA=/usr/local/cuda/lib64/stubs/libcuda.so
[[ -f "$STUB_LIBCUDA" ]] || { echo "with_stub_libcuda.sh: no $STUB_LIBCUDA" >&2; exit 2; }
stub="${TMPDIR:-/tmp}/stub"
mkdir -p "$stub"
ln -sf "$STUB_LIBCUDA" "$stub/libcuda.so.1"
export LD_LIBRARY_PATH="$stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$@"
