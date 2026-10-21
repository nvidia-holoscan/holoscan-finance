#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared configuration for the NYQ equity demo launch and teardown scripts.
set -euo pipefail
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$DEPLOY_DIR/../.." && pwd)"
# Built from deploy/igx_thor/Dockerfile (Holoscan SDK 5.0 + DAQIRI).
RECEIVER_IMAGE="${RECEIVER_IMAGE:-holoscan-finance:hsdk5-daqiri}"
# Helper image with NVIDIA MFT (mstlink, mstreg), iproute2, tcpdump and nsenter; not built here.
MST_IMAGE="${MST_IMAGE:-holoscan-finance:mst}"
APP_NAME="${APP_NAME:-fin-equities-receiver}"
FEED_NAME="${FEED_NAME:-fin-equities-feed}"
OWNER_LABEL="com.nvidia.holoscan-finance.demo"
OWNER_VALUE="${OWNER_VALUE:-nyq-equities}"
NIC_PCIE="${NIC_PCIE:-0004:03:00.0}"
NIC_NAME="${NIC_NAME:-enP4p3s0f0np0}"
NIC_MASTER_CORE="${NIC_MASTER_CORE:-3}"
NIC_RX_CORE="${NIC_RX_CORE:-9}"
LOOPBACK="${LOOPBACK:-0}"
START_FEED="${START_FEED:-$LOOPBACK}"
LOOPBACK_CIDR="${LOOPBACK_CIDR:-192.168.100.1/24}"
LOOPBACK_PEER="${LOOPBACK_PEER:-192.168.100.2}"
FEED_HOST="${FEED_HOST:-}"
if [[ "$LOOPBACK" == 1 ]]; then FEED_HOST="$LOOPBACK_PEER"; fi
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-8080}"
CONTROL_HOST="$WEB_HOST"
if [[ "$CONTROL_HOST" == 0.0.0.0 ]]; then CONTROL_HOST=127.0.0.1; fi
UDP_PORT="${UDP_PORT:-26400}"
TAPE_DIR="${TAPE_DIR:-$REPO_DIR/data/nyq-equities-2025-04-01}"
GPU_UUID="${GPU_UUID:-}"
# daqiri-host: DAQIRI into pinned host memory, Python MITCH decode.
# gpudirect: DAQIRI GPUDirect into GPU memory, CUDA gather + MITCH decode + quote bars.
INGEST="${INGEST:-daqiri-host}"
# Persistent CUDA kernel cache for GPUDirect (avoids ~10 s JIT on every start).
KERNEL_CACHE_DIR="${KERNEL_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/holoscan-finance/cupy}"
CPU_THREADS="${CPU_THREADS:-0}"
SPEED="${SPEED:-10}"
RATE="${RATE:-50000}"
READY_TIMEOUT="${READY_TIMEOUT:-90}"
# Optional full-day button: tcpreplay a captured day (gpu_ingest_bench/capture.sh) at line rate.
FULLDAY_PCAP="${FULLDAY_PCAP:-}"
FULLDAY_BASE_PORT="${FULLDAY_BASE_PORT:-}"
FULLDAY_PACKETS="${FULLDAY_PACKETS:-}"
FULLDAY_WORK="${FULLDAY_WORK:-}"
TCPREPLAY_DIR="${TCPREPLAY_DIR:-}"
TCPREPLAY_LIBS="${TCPREPLAY_LIBS:-}"
REPLAY_NAME="${REPLAY_NAME:-$APP_NAME-replay}"
REPLAY_PORT="${REPLAY_PORT:-8095}"
DRY_RUN=0

fail() { echo "error: $*" >&2; exit 1; }
show() { printf '%q ' "$@"; printf '\n'; }
run() { if [[ "$DRY_RUN" == 1 ]]; then show "$@"; else "$@"; fi; }
# irqbalance runtime ban (irq_isolate.py): keeps interrupts off the DAQIRI cores.
irq_ctl() {
    docker run --rm --privileged --pid=host --entrypoint python3 \
        --mount "type=bind,src=$DEPLOY_DIR/irq_isolate.py,dst=/irq_isolate.py,readonly" \
        "$RECEIVER_IMAGE" /irq_isolate.py "$@"
}
# NVIDIA GPUs that runtime-suspend when idle (Thor's iGPU). While one is suspended, the
# first new CUDA memory allocation in the receiver wakes it (~0.13 s, holding the receiver's
# interpreter lock), so INGEST=gpudirect keeps them powered while the receiver runs.
gpu_pm_auto() {
    local d
    for d in /sys/bus/pci/devices/*; do
        if [[ "$(cat "$d/vendor" 2>/dev/null)" == 0x10de && "$(cat "$d/class" 2>/dev/null)" == 0x03* &&
            "$(cat "$d/power/control" 2>/dev/null)" == auto ]]; then basename "$d"; fi
    done | paste -sd,
}
gpu_pm_set() {  # gpu_pm_set on|auto PCI[,PCI...]
    [[ -n "$2" ]] || return 0
    docker run --rm --privileged --entrypoint sh "$RECEIVER_IMAGE" -c \
        'for d in $(echo "$2" | tr , " "); do echo "$1" > "/sys/bus/pci/devices/$d/power/control"; done' \
        sh "$1" "$2"
}
# systemctl in the host's namespaces. The full-day launch stops fwupd-refresh.timer (an
# hourly firmware-metadata refresh) while it runs; teardown starts it again.
host_systemctl() {
    docker run --rm --privileged --pid=host --entrypoint nsenter "$RECEIVER_IMAGE" \
        -t 1 -m -u -i -n -- systemctl "$@"
}
owned() {
    [[ "$(docker inspect --format "{{index .Config.Labels \"$OWNER_LABEL\"}}" "$1")" == "$OWNER_VALUE" ]]
}
