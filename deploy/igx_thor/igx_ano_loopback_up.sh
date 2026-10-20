#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use only on a dedicated test port. DAQIRI must start before enabling loopback.
# shellcheck source=deploy/igx_thor/ano_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/ano_common.sh"
ACTION=up
case "${1:-}" in
    --check) ACTION=check ;;
    --down) ACTION=down ;;
    --dry-run) DRY_RUN=1 ;;
    "") ;;
    *) fail "usage: $0 [--check|--down|--dry-run]" ;;
esac
[[ $# -le 1 ]] || fail "too many arguments"
# Arguments stay separate from shell code, including overrides with spaces.
run docker run --rm --privileged --net=host --pid=host "$MST_IMAGE" bash -ceu '
    action=$1; pcie=$2; nic=$3; cidr=$4; peer=$5
    [[ "$(basename "$(readlink -f "/sys/class/net/$nic/device")")" == "$pcie" ]] || {
        echo "NIC $nic does not belong to PCIe $pcie" >&2; exit 1;
    }
    if [[ "$action" == check ]]; then
        command -v mstlink >/dev/null
        # Refuse an interface carrying configured IP traffic. The port is
        # reserved for this test, and must start with PHY loopback disabled.
        [[ -z "$(ip -o addr show dev "$nic" scope global)" ]] || {
            echo "PHY loopback requires a dedicated port without global IP addresses" >&2; exit 1;
        }
        [[ -z "$(ip neigh show to "$peer" dev "$nic")" ]] || {
            echo "Loopback peer already has a neighbor entry" >&2; exit 1;
        }
        # mstlink emits ANSI color and reports N/A on a disabled/unplugged port.
        # PPLR is authoritative in that case; all loopback-enable bits must be zero.
        link_info=$(mstlink -d "$pcie" | sed -E "s/\x1B\[[0-9;]*[mK]//g")
        if ! grep -Eq "Loopback Mode[[:space:]]*:[[:space:]]*(No Loopback|Disabled|NO|None)" <<< "$link_info"; then
            local_port=$(cat "/sys/class/net/$nic/dev_port")
            mstreg -d "$pcie" --reg_name PPLR \
                --indexes "local_port=$((local_port + 1)),plane_ind=0,port_type=0,lp_msb=0" --get \
                | grep -Eq "^lb_en[[:space:]]*\|[[:space:]]*0x0+$" || {
                echo "Cannot verify PHY loopback is disabled; inspect mstlink/PPLR output" >&2; exit 1;
            }
        fi
    elif [[ "$action" == up ]]; then
        if [[ "$7" == yes ]]; then
            # Otherwise DHCP activation can delete the temporary test address.
            nsenter -t 1 -m -u -i -n -p -- nmcli device set "$nic" managed no
        fi
        mstlink -d "$pcie" --loopback PH
        mstlink -d "$pcie" --port_state TG
        ip link set "$nic" up
        for ((i=0; i<20; i++)); do
            [[ "$(cat "/sys/class/net/$nic/carrier")" == 1 ]] && break
            sleep 1
        done
        [[ "$(cat "/sys/class/net/$nic/carrier")" == 1 ]] || { echo "No PHY carrier" >&2; exit 1; }
        mac=$(cat "/sys/class/net/$nic/address")
        ip addr add "$cidr" dev "$nic"
        ip neigh add "$peer" lladdr "$mac" nud permanent dev "$nic"
    else
        # Remove only the test address/neighbor. Do not change link speed.
        if [[ -n "$(ip neigh show to "$peer" dev "$nic")" ]]; then
            ip neigh del "$peer" dev "$nic"
        fi
        if ip -o -4 addr show dev "$nic" | awk "{print \$4}" | grep -Fxq "$cidr"; then
            ip addr del "$cidr" dev "$nic"
        fi
        mstlink -d "$pcie" --loopback NO
        if [[ "$7" == yes ]]; then
            nsenter -t 1 -m -u -i -n -p -- nmcli device set "$nic" managed yes
        fi
        if [[ "$6" == 0 ]]; then ip link set "$nic" down; fi
    fi
' ano-loopback "$ACTION" "$ANO_PCIE" "$ANO_NIC" "$LOOPBACK_CIDR" "$LOOPBACK_PEER" "${ORIGINAL_LINK_UP:-1}" "${ORIGINAL_NM_MANAGED:-no}"
