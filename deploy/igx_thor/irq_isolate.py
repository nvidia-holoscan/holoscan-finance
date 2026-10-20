#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep interrupts off the DAQIRI cores through irqbalance's control socket.

irqbalance moves NIC interrupts between CPUs. With the replay sender on the same host,
its transmit-completion softirqs can land on the DPDK RX core and starve the poll loop,
so the NIC queue overflows. ``set CPULIST`` bans those CPUs from IRQ placement until
irqbalance restarts (``set NULL`` clears it); ``get`` prints the banned list, or
``none`` when irqbalance is not running. Needs root and the host's /run: run it in a
privileged container with ``--pid=host``.
"""

import glob
import socket
import sys
import time


def _socket_path():
    for root in ("/proc/1/root", ""):
        found = glob.glob(root + "/run/irqbalance/irqbalance*.sock")
        if found:
            return found[0]
    return None


def _request(path, message):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(path)
        s.sendall(message.encode())
        s.settimeout(2)
        try:
            return s.recv(65536).decode(errors="replace")
        except TimeoutError:
            return ""


def banned(path):
    """irqbalance ``setup`` reports ``... BANNED <hex mask> ...`` (empty while it rescans)."""
    for _ in range(50):
        words = _request(path, "setup").split()
        if "BANNED" in words[:-1]:
            mask = int(words[words.index("BANNED") + 1].replace(",", ""), 16)
            return ",".join(str(c) for c in range(mask.bit_length()) if mask >> c & 1) or "NULL"
        time.sleep(0.1)
    sys.exit("irqbalance did not report its banned CPUs")


def main(argv):
    if not argv or argv[0] not in ("get", "set") or len(argv) != (1 if argv[0] == "get" else 2):
        sys.exit("usage: irq_isolate.py get | set CPULIST|NULL")
    path = _socket_path()
    if path is None:
        print("none")
        return
    if argv[0] == "set":
        want = argv[1]
        if want != "NULL":
            want = ",".join(str(c) for c in sorted({int(c) for c in want.split(",")}))
        _request(path, "settings cpus " + want)
        for _ in range(60):  # applied on irqbalance's next pass (SLEEP, default 10 s)
            if banned(path) == want:
                break
            time.sleep(0.25)
        else:
            sys.exit(f"irqbalance did not apply banned CPUs {want}")
    print(banned(path))


if __name__ == "__main__":
    main(sys.argv[1:])
