# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a deterministic synthetic NYQ day: a tape, its wire capture and a summary.

    python -m holoscan_finance.tools.synth --out DIR [--symbols 500] [--packets 20000]
        [--seconds 600] [--seed 7] [--sender-port 40000] [--drop-every 0]
        [--max-events-per-packet 32] [--reuse]

``DIR/tape/`` is in the ``open_tape`` format, ``DIR/day.pcap`` holds the 4.3 replay
packetization (Ethernet II, IPv4, UDP to 26400) and ``DIR/synth.json`` the arguments, seed,
counts and expected totals. ``--reuse`` keeps an existing directory made with identical
arguments. CPU-only, so every test and the NIC-free validation run without licensed data.
"""

from __future__ import annotations

import argparse


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m holoscan_finance.tools.synth", description=__doc__.splitlines()[0]
    )
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--symbols", type=int, default=500)
    p.add_argument("--packets", type=int, default=20000)
    p.add_argument("--seconds", type=int, default=600, help="tape span after 09:30 New York")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sender-port", type=int, default=40000, help="UDP source port")
    p.add_argument(
        "--drop-every", type=int, default=0, help="leave out every n-th packet (0: none)"
    )
    p.add_argument("--max-events-per-packet", type=int, default=32)
    p.add_argument(
        "--reuse", action="store_true", help="keep DIR if it was made with identical arguments"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser().parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
