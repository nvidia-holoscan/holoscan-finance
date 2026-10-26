#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Replay captured MITCH packets into the SAME receiver at fixed rates, one ingest
# mode at a time, and report loss counters from the NIC, the receiver and the app.
#   BASE_PCAP  capture filtered to one generation's source port (see README)
#   BASE_PORT  that source port;  TCPREPLAY_DIR  directory with tcpreplay/tcprewrite
#   MODES="gpudirect daqiri-host" (add udp for kernel sockets)
#   RATES="4500 10000 20000 50000"  DUR=15 (seconds per rate)
# Leaves the NIC free afterwards; it does not restart any other demo.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; repo="$(cd "$here/../../.." && pwd)"
: "${BASE_PCAP:?}" "${BASE_PORT:?}" "${TCPREPLAY_DIR:?}"
NIC="${NIC_NAME:-enP4p3s0f0np0}"; PCIE="${NIC_PCIE:-0004:03:00.0}"; U="${WEB:-http://127.0.0.1:8080}"
IMAGE="${RECEIVER_IMAGE:-holoscan-finance:hsdk5-daqiri}"; MST="${MST_IMAGE:-holoscan-finance:mst}"; TAPE="${TAPE_DIR:?}"
GPU="${GPU_UUID:-}"; BACKEND="${ANALYTICS_BACKEND:-gpu}"; DUR="${DUR:-15}"
MODES="${MODES:-gpudirect daqiri-host}"; RATES="${RATES:-4500 10000 20000 50000}"; PORTN="${PORT_START:-43000}"
for mode in $MODES; do
    case "$mode" in gpudirect | daqiri-host | udp) ;; *) echo "error: MODES takes gpudirect, daqiri-host or udp, not $mode" >&2; exit 1 ;; esac
done
CACHE="${KERNEL_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/holoscan-finance/cupy}"; mkdir -p "$CACHE"
work="$(mktemp -d)"; export LD_LIBRARY_PATH="${TCPREPLAY_LIBS:-$TCPREPLAY_DIR/../lib/aarch64-linux-gnu}"
root() { docker run --rm --privileged --net=host --pid=host -v "$work:$work" -v "$TCPREPLAY_DIR:$TCPREPLAY_DIR:ro" \
    -e LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$MST" nsenter -t 1 -m -u -n -i -- "$@"; }
ph() { ethtool -S "$NIC" | awk -v k="$1:" '$1==k{print $2}'; }
# DPDK port start resets PHY loopback, so re-apply it after every receiver start.
# Idempotent (replace, not add); remove afterwards with igx_loopback.sh --down.
loopback() {
    docker run --rm --privileged --net=host --pid=host "$MST" bash -ceu '
        pcie=$1; nic=$2; cidr=$3; peer=$4
        mstlink -d "$pcie" --loopback PH >/dev/null
        mstlink -d "$pcie" --port_state TG >/dev/null
        ip link set "$nic" up
        for ((i = 0; i < 20; i++)); do [[ "$(cat "/sys/class/net/$nic/carrier")" == 1 ]] && break; sleep 1; done
        [[ "$(cat "/sys/class/net/$nic/carrier")" == 1 ]] || { echo "No PHY carrier" >&2; exit 1; }
        ip addr replace "$cidr" dev "$nic"
        ip neigh replace "$peer" lladdr "$(cat "/sys/class/net/$nic/address")" nud permanent dev "$nic"
    ' loopback "$PCIE" "$NIC" "${LOOPBACK_CIDR:-192.168.100.1/24}" "${LOOPBACK_PEER:-192.168.100.2}"
}
snap() { curl -s --max-time 10 "$U/snapshot" | python3 -c "
import json,sys; d=json.load(sys.stdin); n=d['network']; l=d.get('live') or {}
print(n['packets'],n['decoded_messages'],n['sequence_gaps'],n['ring_dropped'],n['stale_packets'],n['malformed_packets'],l.get('receive_to_result_ms') or 0)"; }
# Host sysctls the udp mode changes (receive buffer, local delivery and reverse-path filter for
# frames returning through the PHY loopback).
SYSCTLS=(net.core.rmem_max "net.ipv4.conf.$NIC.accept_local" "net.ipv4.conf.$NIC.rp_filter" net.ipv4.conf.all.rp_filter)
saved=()
cleanup() {
    docker logs receiver-bench > "$work/receiver.log" 2>&1 || true; docker rm -f receiver-bench >/dev/null 2>&1 || true
    if ((${#saved[@]})); then root sysctl -q -w "${saved[@]}" || true; fi   # the values found at start
    echo "logs: $work"
}
trap cleanup EXIT
total=$(tcpdump -r "$BASE_PCAP" -n 2>/dev/null | wc -l)
printf "%-11s %8s %9s %9s %10s %6s %9s %6s %6s %10s\n" mode req_pps tx_pkts app_pkts app_msg/s gaps ring_drop stale malf rx2res_ms
for mode in $MODES; do
    extra=(--ingest "$mode" --nic-pcie "$PCIE"); [[ "$mode" == udp ]] && extra=(--ingest udp --udp-host 0.0.0.0)
    docker rm -f receiver-bench >/dev/null 2>&1 || true
    docker run -d --name receiver-bench --runtime nvidia --net host --shm-size 4g --device /dev/infiniband \
        -v /dev/hugepages:/dev/hugepages -v "$repo:/workspace/holoscan-finance:ro" -v "$TAPE:/data/tape:ro" \
        -v "$CACHE:/kernel-cache" -e CUPY_CACHE_DIR=/kernel-cache -w /workspace/holoscan-finance \
        --cap-add IPC_LOCK --cap-add NET_ADMIN --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 \
        -e HOLOSCAN_FINANCE_REPO=/workspace/holoscan-finance -e NVIDIA_DISABLE_REQUIRE=1 -e NVIDIA_VISIBLE_DEVICES=all \
        -e DAQIRI_DV_FLOW_EN=1 -e DAQIRI_RX_FLOW_GROUP=0 ${GPU:+-e CUDA_VISIBLE_DEVICES=$GPU} \
        "$IMAGE" bash deploy/igx_thor/run_python.sh examples/equity_analytics/equity_analytics.py \
        --host 127.0.0.1 --port 8080 --udp-port 26400 --analytics-backend "$BACKEND" "${extra[@]}" >/dev/null
    for _ in $(seq 120); do curl -s --max-time 2 -o /dev/null "$U/feedrate" && break; sleep 2; done
    sleep 10; loopback; sleep 3
    if [[ "$mode" == udp ]]; then
        if ((${#saved[@]} == 0)); then   # record before the first change; cleanup restores these
            for k in "${SYSCTLS[@]}"; do saved+=("$k=$(root sysctl -n "$k")"); done
        fi
        root sysctl -q -w net.core.rmem_max=268435456 "net.ipv4.conf.$NIC.accept_local=1" "net.ipv4.conf.$NIC.rp_filter=0" "net.ipv4.conf.all.rp_filter=0"
    fi
    for rate in $RATES; do
        PORTN=$((PORTN + 1)); n=$((rate * DUR)); ((n > total)) && n=$total
        tcpdump -r "$BASE_PCAP" -c "$n" -w "$work/cut.pcap" 2>/dev/null
        rw=(--portmap="$BASE_PORT:$PORTN" --fixcsum); [[ "$mode" == udp ]] && rw+=(--dstipmap=192.168.100.2/32:192.168.100.1/32)
        "$TCPREPLAY_DIR/tcprewrite" "${rw[@]}" -i "$work/cut.pcap" -o "$work/run.pcap" >/dev/null 2>&1
        docker exec receiver-bench bash deploy/igx_thor/run_python.sh -m holoscan_finance.tools.bench_replay register --port "$PORTN" >/dev/null
        sleep 2; read -r a1 m1 g1 r1 s1 f1 _ <<<"$(snap)"; t1=$(ph tx_packets_phy)
        root "$TCPREPLAY_DIR/tcpreplay" -i "$NIC" --pps="$rate" -K "$work/run.pcap" > "$work/tcpreplay_${mode}_$rate.log" 2>&1
        sleep 8; read -r a2 m2 g2 r2 s2 f2 rr <<<"$(snap)"; t2=$(ph tx_packets_phy)
        secs=$(grep -oE 'sent in [0-9.]+ seconds' "$work/tcpreplay_${mode}_$rate.log" | awk '{print $3}')
        printf "%-11s %8s %9s %9s %10.0f %6s %9s %6s %6s %10.1f\n" "$mode" "$rate" $((t2 - t1)) $((a2 - a1)) \
            "$(python3 -c "print(($m2-$m1)/max(float('${secs:-1}'),1e-3))")" $((g2 - g1)) $((r2 - r1)) $((s2 - s1)) $((f2 - f1)) "$rr"
    done
done
