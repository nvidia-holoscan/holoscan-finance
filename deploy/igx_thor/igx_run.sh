#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launch the NYQ receiver with DAQIRI/ConnectX RX (and optionally the sender,
# PHY loopback and the full-day replay controller). See README.md beside this file.
# shellcheck source=deploy/igx_thor/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ "${1:-}" == --dry-run && $# == 1 ]]; then
    DRY_RUN=1
elif [[ $# != 0 ]]; then
    fail "usage: $0 [--dry-run]; configuration: deploy/igx_thor/README.md"
fi
[[ "$LOOPBACK" =~ ^[01]$ && "$START_FEED" =~ ^[01]$ ]] || fail "LOOPBACK and START_FEED must be 0 or 1"
[[ "$APP_NAME" != "$FEED_NAME" ]] || fail "APP_NAME and FEED_NAME must differ"
# The NIC ingest modes of the app.
case "$INGEST" in
    daqiri-host | gpudirect) MODE=$INGEST ;;
    *) fail "INGEST must be daqiri-host or gpudirect" ;;
esac
[[ "$START_FEED" != 1 || -n "$FEED_HOST" ]] || fail "set FEED_HOST for the local sender, or LOOPBACK=1"
if [[ -n "$FULLDAY_PCAP" ]]; then
    [[ "$MODE" == gpudirect ]] || fail "the full-day button needs INGEST=gpudirect"
    [[ -f "$FULLDAY_PCAP" && "$FULLDAY_BASE_PORT" =~ ^[0-9]+$ && -x "$TCPREPLAY_DIR/tcpreplay" ]] ||
        fail "FULLDAY_PCAP needs FULLDAY_BASE_PORT and TCPREPLAY_DIR (with tcpreplay and tcprewrite)"
    FULLDAY_PCAP="$(cd "$(dirname "$FULLDAY_PCAP")" && pwd)/$(basename "$FULLDAY_PCAP")"
    FULLDAY_WORK="${FULLDAY_WORK:-$(dirname "$FULLDAY_PCAP")/replay_control}"
fi

CONTAINER_REPO=/workspace/holoscan-finance
COMMON=(--net=host --workdir "$CONTAINER_REPO"
    --mount "type=bind,src=$REPO_DIR,dst=$CONTAINER_REPO,readonly"
    -e "HOLOSCAN_FINANCE_REPO=$CONTAINER_REPO"
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1)
ACCEL=(--runtime=nvidia -e NVIDIA_DISABLE_REQUIRE=1 -e NVIDIA_VISIBLE_DEVICES=all
    --device=/dev/infiniband --mount "type=bind,src=/dev/hugepages,dst=/dev/hugepages"
    --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=4g
    --cap-add=NET_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_NICE --cap-add=SYS_PTRACE
    -e "DAQIRI_DV_FLOW_EN=${DAQIRI_DV_FLOW_EN:-1}"
    -e "DAQIRI_RX_FLOW_GROUP=${DAQIRI_RX_FLOW_GROUP:-0}" -e CUDA_DEVICE_ORDER=PCI_BUS_ID)
if [[ -n "$GPU_UUID" ]]; then ACCEL+=(-e "CUDA_VISIBLE_DEVICES=$GPU_UUID"); fi
# Passed through when set: HOLOSCAN_FINANCE_NATIVE_DRAIN=0 selects the Python drain (A/B);
# HOLOSCAN_FINANCE_FAULTS=1 enables POST /debug/hold_gil (fault injection, tests only).
for var in HOLOSCAN_FINANCE_NATIVE_DRAIN HOLOSCAN_FINANCE_FAULTS; do
    if [[ -n "${!var:-}" ]]; then ACCEL+=(-e "$var=${!var}"); fi
done
if [[ "$MODE" == gpudirect ]]; then
    ACCEL+=(--mount "type=bind,src=$KERNEL_CACHE_DIR,dst=/kernel-cache" -e CUPY_CACHE_DIR=/kernel-cache)
fi
APP_ARGS=(applications/equity_analytics/equity_analytics.py --ingest "$MODE"
    --host "$WEB_HOST" --port "$WEB_PORT" --udp-port "$UDP_PORT"
    --nic-pcie "$NIC_PCIE" --nic-master-core "$NIC_MASTER_CORE" --nic-rx-core "$NIC_RX_CORE"
    --cpu-threads "$CPU_THREADS" --analytics-backend "${ANALYTICS_BACKEND:-compare}")
if [[ -n "$FULLDAY_PCAP" ]]; then APP_ARGS+=(--replay-control "http://127.0.0.1:$REPLAY_PORT"); fi
LABELS=(--label "$OWNER_LABEL=$OWNER_VALUE" --label "hf.loopback=$LOOPBACK"
    --label "hf.pcie=$NIC_PCIE" --label "hf.nic=$NIC_NAME" --label "hf.cidr=$LOOPBACK_CIDR"
    --label "hf.peer=$LOOPBACK_PEER" --label "hf.mst-image=$MST_IMAGE" --label "hf.feed=$FEED_NAME"
    --label "hf.ingest=$INGEST" --label "hf.replay=${FULLDAY_PCAP:+$REPLAY_NAME}")
GPU_PM=""
if [[ "$MODE" == gpudirect ]]; then GPU_PM=$(gpu_pm_auto); fi
LABELS+=(--label "hf.gpu-pm=$GPU_PM")
FEED=(docker run -d --name "$FEED_NAME" --label "$OWNER_LABEL=$OWNER_VALUE"
    "${COMMON[@]}" --mount "type=bind,src=$TAPE_DIR,dst=/data/tape,readonly"
    "$RECEIVER_IMAGE" bash deploy/igx_thor/run_python.sh -m holoscan_finance.tools.replay_sender
    --host "$FEED_HOST" --port "$UDP_PORT" --data /data/tape
    --loops 0 --control-url "http://$CONTROL_HOST:$WEB_PORT")
# Privileged: tcpreplay writes raw frames on the NIC. Paths keep their host names.
REPLAY=(docker run -d --name "$REPLAY_NAME" --label "$OWNER_LABEL=$OWNER_VALUE" --privileged
    "${COMMON[@]}" --mount "type=bind,src=$TAPE_DIR,dst=/data/tape,readonly")
for dir in "$(dirname "${FULLDAY_PCAP:-/}")" "$TCPREPLAY_DIR" "$TCPREPLAY_LIBS" "$FULLDAY_WORK"; do
    if [[ -n "$dir" && "$dir" != / ]]; then REPLAY+=(--mount "type=bind,src=$dir,dst=$dir"); fi
done
REPLAY+=("$RECEIVER_IMAGE" bash deploy/igx_thor/run_python.sh deploy/igx_thor/gpu_ingest_bench/replay_control.py
    --pcap "$FULLDAY_PCAP" --base-port "$FULLDAY_BASE_PORT" --nic "$NIC_NAME"
    --tcpreplay-dir "$TCPREPLAY_DIR" --tcpreplay-libs "$TCPREPLAY_LIBS" --work "$FULLDAY_WORK"
    --app "http://$CONTROL_HOST:$WEB_PORT" --tape /data/tape --port "$REPLAY_PORT"
    --avoid-cores "$NIC_MASTER_CORE,$NIC_RX_CORE"
    ${FULLDAY_PACKETS:+--packets "$FULLDAY_PACKETS"})
CHECK=(docker run --rm "${COMMON[@]}" "${ACCEL[@]}" "$RECEIVER_IMAGE"
    bash deploy/igx_thor/run_python.sh deploy/igx_thor/preflight.py check --pcie "$NIC_PCIE"
    --master-core "$NIC_MASTER_CORE" --rx-core "$NIC_RX_CORE" --ingest "$MODE")

if [[ "$DRY_RUN" == 0 ]]; then
    command -v docker >/dev/null || fail "docker is required"
    command -v python3 >/dev/null || fail "python3 is required"
    docker image inspect "$RECEIVER_IMAGE" >/dev/null || fail "build RECEIVER_IMAGE first (deploy/igx_thor/README.md)"
    if [[ "$MODE" == gpudirect ]]; then mkdir -p "$KERNEL_CACHE_DIR"; fi
    if [[ -n "$FULLDAY_PCAP" ]]; then
        mkdir -p "$FULLDAY_WORK"
        FWUPD_TIMER=$(host_systemctl is-active fwupd-refresh.timer || true)
        LABELS+=(--label "hf.fwupd-timer=$FWUPD_TIMER")
    fi
    for name in "$APP_NAME" "$FEED_NAME" ${FULLDAY_PCAP:+"$REPLAY_NAME"}; do
        if docker container inspect "$name" >/dev/null 2>&1; then
            fail "container $name already exists; inspect it or use igx_down.sh for this demo"
        fi
    done
    if [[ "$START_FEED" == 1 ]]; then
        [[ -f "$TAPE_DIR/manifest.json" && -f "$TAPE_DIR/events.bin" ]] || fail "prepare NYQ data first; expected manifest.json and events.bin in $TAPE_DIR"
    fi
    if [[ "$LOOPBACK" == 1 ]]; then
        docker image inspect "$MST_IMAGE" >/dev/null || fail "MST_IMAGE is required for PHY loopback"
        ORIGINAL_NM_MANAGED=no
        if command -v nmcli >/dev/null; then
            ORIGINAL_NM_MANAGED=$(nmcli -g GENERAL.NM-MANAGED device show "$NIC_NAME")
            [[ "$ORIGINAL_NM_MANAGED" =~ ^(yes|no)$ ]] || fail "cannot read NetworkManager state for $NIC_NAME"
        fi
        export ORIGINAL_NM_MANAGED
        LABELS+=(--label "hf.nm-managed=$ORIGINAL_NM_MANAGED")
        # Read-only check BEFORE the receiver starts or NIC settings change.
        bash "$DEPLOY_DIR/igx_loopback.sh" --check
        link_flags=$(cat "/sys/class/net/$NIC_NAME/flags")
        LABELS+=(--label "hf.link-up=$((link_flags & 1))")
        # The sender shares this host: its NIC interrupts must stay off the DAQIRI cores.
        IRQ_BANNED=$(irq_ctl get)
        LABELS+=(--label "hf.irq-banned=$IRQ_BANNED")
    fi
fi
run "${CHECK[@]}"

created=0
cleanup() {
    local status=$?
    if [[ "$status" != 0 && "$created" == 1 ]]; then
        docker logs --tail 30 "$APP_NAME" >&2 || true
        APP_NAME="$APP_NAME" bash "$DEPLOY_DIR/igx_down.sh" || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run docker run -d --name "$APP_NAME" "${LABELS[@]}" "${COMMON[@]}" "${ACCEL[@]}" \
    "$RECEIVER_IMAGE" bash deploy/igx_thor/run_python.sh "${APP_ARGS[@]}"
if [[ "$DRY_RUN" == 0 ]]; then created=1; fi
run gpu_pm_set on "$GPU_PM"
if [[ "${FWUPD_TIMER:-}" == active ]]; then run host_systemctl stop fwupd-refresh.timer; fi
run python3 "$DEPLOY_DIR/preflight.py" wait --url "http://$CONTROL_HOST:$WEB_PORT" \
    --timeout "$READY_TIMEOUT" --speed "$SPEED" --rate "$RATE"
if [[ "$LOOPBACK" == 1 ]]; then
    if [[ "$DRY_RUN" == 1 ]]; then
        bash "$DEPLOY_DIR/igx_loopback.sh" --dry-run
    else
        bash "$DEPLOY_DIR/igx_loopback.sh"
    fi
    IRQ_BANNED="${IRQ_BANNED:-NULL}"  # dry run: show the ban command
    if [[ "$IRQ_BANNED" != none ]]; then
        ban=$({ [[ "$IRQ_BANNED" == NULL ]] || tr , '\n' <<< "$IRQ_BANNED"
                printf '%s\n' "$NIC_MASTER_CORE" "$NIC_RX_CORE"; } | sort -nu | paste -sd,)
        run irq_ctl set "$ban"
    fi
fi
if [[ "$START_FEED" == 1 ]]; then run "${FEED[@]}"; fi
if [[ -n "$FULLDAY_PCAP" ]]; then run "${REPLAY[@]}"; fi
printf 'NYQ equity demo: http://127.0.0.1:%s/ (%s RX, UDP destination port %s)\n' "$WEB_PORT" "$MODE" "$UDP_PORT"
printf 'Stop: APP_NAME=%q bash %q/igx_down.sh\n' "$APP_NAME" "$DEPLOY_DIR"
