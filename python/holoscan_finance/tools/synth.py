# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a deterministic synthetic equity day: a tape, its wire capture and a summary.

    python -m holoscan_finance.tools.synth --out DIR [--symbols 500] [--packets 20000]
        [--seconds 600] [--seed 7] [--sender-port 40000] [--drop-every 0]
        [--max-events-per-packet 30] [--encoder numpy|gpu] [--reuse]

``DIR/tape/`` is in the ``open_tape`` format, ``DIR/day.pcap`` holds the replay
sender's packetization (Ethernet II, IPv4, UDP to 26400) and ``DIR/synth.json`` the arguments, seed,
counts and expected totals (``capture``: what a lossless receiver of the pcap counts).
``--reuse`` keeps an existing directory made with identical arguments. The tape is generated
on the CPU; ``--encoder gpu`` packs and frames the capture on the GPU (CuPy), writing the
same bytes as the NumPy packer. The outputs support tests and NIC-free validation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..nyq.mapping import MAX_PACKET_EVENTS
from ..nyq.synthetic import ENCODERS, synthetic_capture, synthetic_tape

_FILES = ("tape/events.bin", "tape/manifest.json", "day.pcap")


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
    p.add_argument("--max-events-per-packet", type=int, default=MAX_PACKET_EVENTS)
    p.add_argument(
        "--encoder", choices=ENCODERS, default="numpy", help="pack and frame the capture with"
    )
    p.add_argument("--gpu-device", type=int, default=0, help="CUDA device of --encoder gpu")
    p.add_argument(
        "--reuse", action="store_true", help="keep DIR if it was made with identical arguments"
    )
    return p


def _size(path: Path) -> int | None:
    return path.stat().st_size if path.is_file() else None


def _reusable(out: Path, args: dict) -> bool:
    """``out`` was completed by a run with ``args`` and its files are unchanged in size."""
    try:
        previous = json.loads((out / "synth.json").read_text())
    except (OSError, ValueError):
        return False
    files = previous.get("files", {})
    return previous.get("args") == args and all(
        _size(out / name) == files.get(name) for name in _FILES
    )


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(a.out)
    # The encoder does not change the bytes, so either one may reuse the other's output.
    args = {
        k: v
        for k, v in sorted(vars(a).items())
        if k not in ("out", "reuse", "encoder", "gpu_device")
    }
    if a.reuse and _reusable(out, args):
        print(f"synth: {out} already holds this data set (--reuse)")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    # Written last, so an interrupted run never looks reusable.
    (out / "synth.json").unlink(missing_ok=True)
    try:
        tape = synthetic_tape(
            out / "tape",
            symbols=a.symbols,
            packets=a.packets,
            seconds=a.seconds,
            max_events_per_packet=a.max_events_per_packet,
            seed=a.seed,
        )
        capture = synthetic_capture(
            out / "tape",
            out / "day.pcap",
            sender_port=a.sender_port,
            max_events_per_packet=a.max_events_per_packet,
            drop_every=a.drop_every,
            encoder=a.encoder,
            gpu_device=a.gpu_device,
        )
    except ValueError as exc:
        raise SystemExit(f"synth: {exc}") from exc
    summary = {
        "args": args,
        "tape": {
            "date": tape["date"],
            "symbols": len(tape["symbols"]),
            "events": tape["events"],
            **tape["kinds"],
            "start_ns": tape["start_ns"],
            "end_ns": tape["end_ns"],
        },
        "capture": capture,
        "files": {name: _size(out / name) for name in _FILES},
    }
    (out / "synth.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"out": str(out), **capture}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
