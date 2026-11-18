# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture full-run p50/p99 while the existing Holoscan receiver and sender run.

    python -m holoscan_finance.tools.measure --output RUN.json [--url http://127.0.0.1:8080]
        [--seconds 120] [--rate 75000] [--seek 50400]

Wait for a complete, validated window of a fresh generation, record every packet and
calculation of a ``--seconds`` capture through the dashboard's ``/capture`` endpoints, and
save it with the counters around it; separate CPU and GPU captures are then compared by
``compare_runs``. It knows the NIC-free ``replay`` front end, whose pace and position are
the receiver's CLI (``/feedrate`` does not reach it), so there any complete window starts
the capture. CPU only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from .replay_sender import request

TRANSPORT_SCOPE = {  # by /snapshot network.ingest
    "udp": "Linux kernel UDP sockets",
    "daqiri-host": "DAQIRI/ConnectX host-memory reception",
    "gpudirect": "DAQIRI/ConnectX GPUDirect reception into GPU memory",
    "replay": "pcap replay from GPU memory (software NIC)",
}


def measure(
    base: str, output: str, *, seconds: float = 120, rate: float = 75000, seek: float = 50400
) -> dict:
    before = request(base, "/snapshot")
    backend = before.get("analytics_backend", "compare")
    replay = before["network"]["ingest"] == "replay"
    original_control = {k: before["feed_control"][k] for k in ("speed", "rate", "paused")}
    target = request(base, "/feedrate", dict(speed=0, rate=rate, seek=seek, paused=False))
    try:
        deadline = time.monotonic() + 180
        while True:
            state = request(base, "/snapshot")
            if state.get("error"):
                raise RuntimeError(state["error"])
            live, cpu = state.get("live"), state.get("comparison")
            ready = (
                bool(
                    cpu
                    and cpu["matched_work"]
                    and cpu["agreement"]
                    and live
                    and cpu["timestamp_ns"] == live["timestamp_ns"]
                )
                if backend == "compare"
                else bool(live)
            )
            # The seek must have reached the sender, which then registered a new generation;
            # replay mode has no external sender and ignores /feedrate, so any window counts.
            fresh = replay or (
                state["generation"] != before["generation"]
                and (state.get("sender_status") or {}).get("seek_seq") == target["seek_seq"]
            )
            if (
                fresh
                and live
                and ready
                and live["bars"]
                == state["settings"]["window_seconds"] // state["settings"]["bar_seconds"]
                and live["eligible"] >= 2
            ):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("replay did not produce a complete valid window in 180 seconds")
            time.sleep(1)
        start_network = state["network"]
        capture = request(base, "/capture", {"seconds": seconds})
        print(
            f"Capturing all packets and calculations for {seconds}s at requested {rate:,} observations/s",
            flush=True,
        )
        # Drain late computations after the receiver-timestamp interval closes.
        while time.time_ns() < capture["end_ns"] + 2_000_000_000:
            time.sleep(1)
        result = request(base, "/capture/finish", {})
        after = request(base, "/snapshot")
        result["created_utc"] = datetime.now(UTC).isoformat()
        result["requested_rate"] = rate
        result["seek_seconds_utc"] = seek
        result["source_date"] = state["profile"]["date"]
        result["transport_scope"] = TRANSPORT_SCOPE[state["network"]["ingest"]]
        result["network_counters_before"] = {
            k: v for k, v in start_network.items() if k != "buckets"
        }
        result["network_counters_after"] = {
            k: v for k, v in after["network"].items() if k != "buckets"
        }
        result["counter_note"] = (
            "Counter snapshots surround capture and include the two-second completion grace period. Percentiles and throughput use only receiver timestamps inside the exact capture interval."
        )
        result["latency_scope"] = (
            "GPU/CPU result latency is measured once per calculated window from its triggering packet, excluding browser refresh. Capture begins after a full return window and startup warm-up; includes any periodic rebuilds occurring during capture."
        )
        result["cpu_affinity_count"] = (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        )
        worker = after["comparison"] if backend == "compare" else after["live"]
        result["analytics_workers"] = {
            key: worker.get(key)
            for key in (
                "cpu_threads",
                "cpu_affinity",
                "cpu_thread_trials_ms",
                "cpu_tuned_bars",
                "gpu_name",
                "blas",
            )
        }
        if backend != "compare":
            result["analytics_workers"]["cpu_threads"] = worker.get("threads")
        result["cpu_affinity_count_note"] = (
            "cpu_affinity_count describes the capture client; analytics_workers.cpu_affinity describes the actual CPU worker."
        )
        for key in ("ring_dropped", "malformed_packets", "out_of_order_packets"):
            if after["network"][key] != start_network[key]:
                result["valid"] = False
        if after.get("error"):
            result["valid"] = False
            result["error"] = after["error"]
        if not result["valid"] and backend == "compare":
            result["matched_analytics_speedup"] = None
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "valid",
                        "observations",
                        "received_observations_per_second",
                        "gpu_receive_to_result_ms",
                        "gpu_analytics_ms",
                        "cpu_analytics_ms",
                        "matched_analytics_speedup",
                        "analytics_backend",
                        "analytics_ms",
                        "receive_to_result_ms",
                    )
                    if k in result
                },
                indent=2,
            ),
            flush=True,
        )
        if not result["valid"]:
            raise RuntimeError(f"capture failed validation; inspect {path}")
        return result
    finally:
        request(base, "/feedrate", original_control)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m holoscan_finance.tools.measure", description=__doc__.splitlines()[0]
    )
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--rate", type=float, default=75000)
    p.add_argument(
        "--seek", type=float, default=50400, help="seconds since UTC midnight, default 14:00"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    measure(args.url, args.output, seconds=args.seconds, rate=args.rate, seek=args.seek)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
