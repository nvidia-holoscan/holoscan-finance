#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Capture the synthetic replay sender's MITCH wire packets (egress on the ConnectX port) from
# a fresh replay generation, so they can be replayed at arbitrary rates.
# Requires a running demo (receiver + feed) and the MST_IMAGE helper image (tcpdump, nsenter).
set -euo pipefail
NIC="${NIC_NAME:-enP4p3s0f0np0}"; UDP_PORT="${UDP_PORT:-26400}"; WEB="${WEB:-http://127.0.0.1:8080}"
OUT="${OUT:?set OUT=/path/capture.pcap}"; PACKETS="${PACKETS:-3000000}"; SEEK="${SEEK:-48600}"
MST_IMAGE="${MST_IMAGE:-holoscan-finance:mst}"
dir="$(cd "$(dirname "$OUT")" && pwd)"
docker run -d --name holofin-capture --privileged --net=host --pid=host "$MST_IMAGE" \
    nsenter -t 1 -m -u -n -i -- tcpdump -i "$NIC" -Q out -n -s 0 -B 262144 -c "$PACKETS" \
    -w "$dir/$(basename "$OUT")" "udp dst port $UDP_PORT" >/dev/null
trap 'curl -s -X POST "$WEB/feedrate" -H "content-type: application/json" -d "{\"rate\": 75000, \"speed\": 0}" >/dev/null; docker rm -f holofin-capture >/dev/null 2>&1 || true' EXIT
sleep 5
# A seek starts a new generation from sequence 1; the control API caps rate at 200k.
curl -s -X POST "$WEB/feedrate" -H 'content-type: application/json' -d "{\"seek\": $SEEK, \"rate\": 200000, \"speed\": 0}"
echo; echo "capturing $PACKETS packets into $OUT ..."
docker wait holofin-capture >/dev/null
echo "done. Next: PYTHONPATH=python python3 -m holoscan_finance.tools.bench_replay inspect $OUT"
