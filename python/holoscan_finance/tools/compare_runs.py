# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare separately executed CPU/GPU streaming captures on predefined windows.

    python -m holoscan_finance.tools.compare_runs --cpu CPU.json --gpu GPU.json --output OUT.json
        [--start ISO --end ISO | --all-shared]

From 4.3 ``equities.compare_runs`` (same flags). A speedup is claimed only when both
``measure`` captures cover every predefined window with identical inputs (SHA-256 of the
window), numerically agreeing results and the same incremental/rebuild work; one missing,
duplicated or disagreeing window voids it. CPU only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from ..analytics.agreement import agreement
from ..telemetry.capture import percentiles


def shared_interval(cpu: dict, gpu: dict) -> tuple[int, int]:
    """Full overlap of the two saved time ranges; compare_runs rejects holes."""
    stamps = [[r["bar_ns"] for r in run["single_calculations"]] for run in (cpu, gpu)]
    if not all(stamps):
        raise ValueError("both captures need calculated windows")
    start = max(min(s) for s in stamps)
    end = min(max(s) for s in stamps) + cpu["settings"]["bar_seconds"] * 10**9
    if start >= end:
        raise ValueError("captures have no shared historical interval")
    return start, end


def compare_runs(cpu: dict, gpu: dict, start_ns: int, end_ns: int) -> dict:
    step = cpu["settings"]["bar_seconds"] * 10**9
    expected = list(range(start_ns, end_ns, step))
    errors = []
    if not expected or start_ns % step or end_ns % step:
        raise ValueError("choose a nonempty interval aligned with the return-bar boundaries")
    if cpu.get("analytics_backend") != "cpu" or gpu.get("analytics_backend") != "gpu":
        raise ValueError("provide CPU-only then GPU-only streaming captures")
    for key in (
        "settings",
        "source_date",
        "requested_rate",
        "ingest",
        "minimum_update_interval_seconds",
    ):
        if cpu[key] != gpu[key]:
            errors.append(f"different {key}")
    if cpu.get("wire_protocol") != gpu.get("wire_protocol"):
        errors.append("different wire_protocol")
    if not cpu["valid"] or not gpu["valid"]:
        errors.append("a capture failed transport/completion checks")
    maps = []
    for name, run in (("cpu", cpu), ("gpu", gpu)):
        rows = [r for r in run["single_calculations"] if start_ns <= r["bar_ns"] < end_ns]
        mapped = {r["bar_ns"]: r for r in rows}
        if len(mapped) != len(rows) or sorted(mapped) != expected:
            errors.append(f"{name}: missing or duplicate predefined windows")
        maps.append(mapped)
    paired = []
    for stamp in expected:
        if stamp not in maps[0] or stamp not in maps[1]:
            continue
        a, b = maps[0][stamp], maps[1][stamp]
        same = a["input_sha256"] == b["input_sha256"]
        agrees = agreement(a["validation"], b["validation"])
        if not same or not agrees:
            errors.append(f"window {stamp}: inputs_match={same}, numerical_agreement={agrees}")
        paired.append(
            dict(
                bar_ns=stamp,
                input_sha256=a["input_sha256"],
                inputs_match=same,
                numerical_agreement=agrees,
                eligible=a["eligible"],
                cpu_ms=a["analytics_ms"],
                gpu_ms=b["analytics_ms"],
                cpu_method=a["method"],
                gpu_method=b["method"],
                cpu_receive_to_result_ms=a["receive_to_result_ms"],
                gpu_receive_to_result_ms=b["receive_to_result_ms"],
            )
        )
    work = []
    for mapped in maps:
        work.append(
            dict(
                methods=dict(Counter(r["method"] for r in mapped.values())),
                added_bars=sum(r["added_bars"] for r in mapped.values()),
                expired_bars=sum(r["expired_bars"] for r in mapped.values()),
            )
        )
    if work[0] != work[1]:
        errors.append("different total incremental/rebuild work in the predefined interval")
    cpu_stats = percentiles([r["cpu_ms"] for r in paired])
    gpu_stats = percentiles([r["gpu_ms"] for r in paired])
    return dict(
        methodology="Separate CPU-only and GPU-only analytics runs with concurrent streaming ingestion; compare all predefined historical windows. No overlapping analytics backends within either run.",
        timing_scope="Same rolling float64 analytics, required GPU transfers and synchronization; excludes process queues, network ingestion and browser. These enter separately reported packet-to-result latency where applicable.",
        interval=dict(start_ns=start_ns, end_ns_exclusive=end_ns, expected_windows=len(expected)),
        source_date=cpu["source_date"],
        wire_protocol=cpu.get("wire_protocol"),
        settings=cpu["settings"],
        requested_rate=cpu["requested_rate"],
        ingest=cpu["ingest"],
        valid=not errors,
        errors=errors,
        cpu_analytics_ms=cpu_stats,
        gpu_analytics_ms=gpu_stats,
        median_analytics_speedup=cpu_stats["p50"] / gpu_stats["p50"] if not errors else None,
        cpu_receive_to_result_ms=percentiles([r["cpu_receive_to_result_ms"] for r in paired]),
        gpu_receive_to_result_ms=percentiles([r["gpu_receive_to_result_ms"] for r in paired]),
        cpu_work=work[0],
        gpu_work=work[1],
        matched_calculations=paired,
        runs={
            name: {
                k: run[k]
                for k in (
                    "created_utc",
                    "seconds",
                    "observations",
                    "received_observations_per_second",
                    "analytics_workers",
                    "network_counters_before",
                    "network_counters_after",
                    "transport_scope",
                    "valid",
                )
            }
            for name, run in (("cpu", cpu), ("gpu", gpu))
        },
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m holoscan_finance.tools.compare_runs", description=__doc__.splitlines()[0]
    )
    p.add_argument("--cpu", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument(
        "--all-shared",
        action="store_true",
        help="compare the entire common historical interval, rejecting missing windows",
    )
    p.add_argument("--output", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    a = p.parse_args(argv)
    cpu, gpu = (json.loads(Path(path).read_text()) for path in (a.cpu, a.gpu))
    if a.all_shared:
        if a.start or a.end:
            p.error("--all-shared cannot be combined with --start or --end")
        start, end = shared_interval(cpu, gpu)
    else:
        start, end = (
            int(datetime.fromisoformat(t).timestamp()) * 10**9
            for t in (a.start or "2025-04-01T14:10:00+00:00", a.end or "2025-04-01T14:25:00+00:00")
        )
    report = compare_runs(cpu, gpu, start, end)
    report["window_selection"] = (
        "Entire common historical interval; no missing windows permitted"
        if a.all_shared
        else "Specified historical interval"
    )
    Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("runs", "matched_calculations")}, indent=2
        )
    )
    if not report["valid"]:
        raise SystemExit("Separate streaming runs failed comparison; no speedup claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
