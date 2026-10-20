# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for replaying captured MITCH packets into the receiver at controlled rates.

The Python replay sender is limited to a few thousand packets per second, so the
receiver's ceiling is measured by capturing the sender's own wire packets once
(``deploy/igx_thor/gpu_ingest_bench/capture.sh``) and replaying them with
``tcpreplay``. The receiver only accepts packets from the UDP source port named in
the active replay manifest, and each generation needs a fresh port:

    inspect   list source ports in a capture; verify one port's MITCH sequence continuity
    register  post a new generation for a given source port (same payload replay.py sends)
"""

from __future__ import annotations

import argparse
import collections
import struct
import time


def inspect(path, port=None):
    from holoscan.marketdata import mitch

    order, counts = [], collections.Counter()
    first = nxt = None
    packets = messages = discontinuities = 0
    with open(path, "rb") as f:
        f.read(24)
        while True:
            h = f.read(16)
            if len(h) < 16:
                break
            pkt = f.read(struct.unpack("<IIII", h)[2])
            sport = struct.unpack(">H", pkt[34:36])[0]
            counts[sport] += 1
            if not order or order[-1] != sport:
                order.append(sport)
            if port is not None and sport == port:
                _, seq, msgs = mitch.parse_unit(pkt[42:])
                if first is None:
                    first = seq
                elif seq != nxt:
                    discontinuities += 1
                nxt = seq + len(msgs)
                packets += 1
                messages += len(msgs)
    out = {"ports_in_order": order, "packets_per_port": dict(counts)}
    if port is not None:
        out.update(
            port=port,
            packets=packets,
            first_sequence=first,
            discontinuities=discontinuities,
            messages=messages,
        )
    return out


def register(port, url, tape):
    from .data import open_tape
    from .mitch import PROTOCOL, TIMEZONE
    from .replay import request

    manifest, _ = open_tape(tape)
    payload = {
        **manifest,
        "generation": time.time_ns(),
        "wire_protocol": PROTOCOL,
        "market_timezone": TIMEZONE,
        "sender_port": int(port),
    }
    return request(url, "/manifest", payload)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("inspect")
    i.add_argument("pcap")
    i.add_argument("--port", type=int)
    r = sub.add_parser("register")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--tape", default="/data/tape")
    a = p.parse_args()
    print(inspect(a.pcap, a.port) if a.command == "inspect" else register(a.port, a.url, a.tape))


if __name__ == "__main__":
    main()
