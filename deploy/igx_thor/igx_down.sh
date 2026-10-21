#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Teardown uses the settings recorded on the receiver, even in a new shell.
# shellcheck source=deploy/igx_thor/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# == 0 ]] || fail "usage: APP_NAME=fin-equities-receiver $0"
if ! docker container inspect "$APP_NAME" >/dev/null 2>&1; then
    echo "No receiver container $APP_NAME; no changes made."
    exit 0
fi
owned "$APP_NAME" || fail "refusing to stop unrelated container $APP_NAME"
label() { docker inspect --format "{{index .Config.Labels \"hf.$1\"}}" "$APP_NAME"; }
# Host-setting restores run in the receiver's own image.
RECEIVER_IMAGE=$(docker inspect --format '{{.Config.Image}}' "$APP_NAME")
LOOPBACK=$(label loopback)
FEED_NAME=$(label feed)
if docker container inspect "$FEED_NAME" >/dev/null 2>&1; then
    owned "$FEED_NAME" || fail "refusing to stop unrelated container $FEED_NAME"
    docker rm -f "$FEED_NAME"
fi
REPLAY_NAME=$(label replay)
if [[ -n "$REPLAY_NAME" ]] && docker container inspect "$REPLAY_NAME" >/dev/null 2>&1; then
    owned "$REPLAY_NAME" || fail "refusing to stop unrelated container $REPLAY_NAME"
    docker rm -f "$REPLAY_NAME"
fi
# Keep the stopped container and its settings until NIC cleanup succeeds.
# A failed cleanup can then be retried with this same command.
docker stop --time 15 "$APP_NAME"
if [[ "$LOOPBACK" == 1 ]]; then
    ORIGINAL_LINK_UP=$(label link-up); export ORIGINAL_LINK_UP
    ORIGINAL_NM_MANAGED=$(label nm-managed); export ORIGINAL_NM_MANAGED
    NIC_PCIE=$(label pcie); export NIC_PCIE
    NIC_NAME=$(label nic); export NIC_NAME
    LOOPBACK_CIDR=$(label cidr); export LOOPBACK_CIDR
    LOOPBACK_PEER=$(label peer); export LOOPBACK_PEER
    MST_IMAGE=$(label mst-image); export MST_IMAGE
    bash "$DEPLOY_DIR/igx_loopback.sh" --down
    IRQ_BANNED=$(label irq-banned)
    if [[ -n "$IRQ_BANNED" && "$IRQ_BANNED" != none ]]; then
        irq_ctl set "$IRQ_BANNED" >/dev/null ||
            echo "warning: could not restore irqbalance banned CPUs ($IRQ_BANNED)" >&2
    fi
fi
GPU_PM=$(label gpu-pm)
if [[ -n "$GPU_PM" ]]; then
    gpu_pm_set auto "$GPU_PM" || echo "warning: could not restore runtime PM on $GPU_PM" >&2
fi
if [[ "$(label fwupd-timer)" == active ]]; then
    host_systemctl start fwupd-refresh.timer ||
        echo "warning: could not restart fwupd-refresh.timer" >&2
fi
docker rm "$APP_NAME"
echo "NYQ equity demo stopped."
