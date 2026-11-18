# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for replaying captured SBE packets into the receiver at controlled rates.

The Python replay sender is limited to a few thousand packets per second, so the
receiver's ceiling is measured by capturing the sender's own wire packets once
(``deploy/igx_thor/gpu_ingest_bench/capture.sh``) and replaying them with
``tcpreplay``. The receiver only accepts packets from the UDP source port named in
the active replay manifest, and each generation needs a fresh port:

    inspect   list source ports in a capture; verify one port's SBE sequence continuity
    register  post a new generation for a given source port (same payload replay_sender sends)

Source ports and payloads are found from each frame's IPv4 header length, so frames with
IP options parse correctly. CPU only.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ..equity.mapping import PROTOCOL, TIMEZONE
from ..equity.tape import open_tape
from ..marketdata import sbe_codec as sbe
from ..marketdata.framing import PcapFrames, strip_udp_datagram
from .replay_sender import request


def inspect(path: str, port: int | None = None) -> dict:
    """Source ports in order of appearance and packets per port; with ``port``, that port's
    first sequence, sequence discontinuities and SBE entry count."""
    frames = PcapFrames.read(path)
    rel = frames.udp_src_port_offsets()
    udp = np.flatnonzero(rel >= 0)
    at = frames.offsets[udp] + rel[udp]
    ports = frames.data[at].astype(np.int64) << 8 | frames.data[at + 1]
    runs = np.flatnonzero(np.r_[True, ports[1:] != ports[:-1]]) if len(ports) else []
    values, counts = np.unique(ports, return_counts=True)
    out = {
        "ports_in_order": ports[runs].tolist(),
        "packets_per_port": dict(zip(values.tolist(), counts.tolist(), strict=True)),
    }
    if port is None:
        return out
    first = nxt = None
    packets = messages = discontinuities = 0
    for i in udp[ports == port]:
        got = strip_udp_datagram(frames.frame(int(i)))
        if got is None:  # a fragment or a truncated datagram carries no whole unit
            continue
        seq, entries = sbe.parse_packet(got[0])
        if first is None:
            first = seq
        elif seq != nxt:
            discontinuities += 1
        nxt = seq + len(entries)
        packets += 1
        messages += len(entries)
    out.update(
        port=port,
        packets=packets,
        first_sequence=first,
        discontinuities=discontinuities,
        messages=messages,
    )
    return out


def register(port: int, url: str, tape: str) -> dict:
    """Register a new generation of ``tape`` sent from UDP source port ``port``."""
    manifest, _ = open_tape(tape)
    payload = {
        **manifest,
        "generation": time.time_ns(),
        "wire_protocol": PROTOCOL,
        "market_timezone": TIMEZONE,
        "sender_port": int(port),
    }
    return request(url, "/manifest", payload)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m holoscan_finance.tools.bench_replay",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("inspect")
    i.add_argument("pcap")
    i.add_argument("--port", type=int)
    r = sub.add_parser("register")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--tape", default="/data/tape")
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    print(inspect(a.pcap, a.port) if a.command == "inspect" else register(a.port, a.url, a.tape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
