# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``/snapshot`` JSON, built from host objects only.

The bars lane publishes exact counters per batch and a symbol snapshot at most every
250 ms, so a snapshot never touches CUDA, never takes a data-plane lock, and reads the
directory without a lock. ``snapshot_json`` encodes it with ``json.dumps`` (strict: no NaN,
no NumPy scalars) and reuses the encoding of the two large blocks, the live result (per
published result) and the universe (per directory), so a request holds the interpreter
lock for about a millisecond at 2,886 symbols. ``index.html``, ``measure.py``
and ``compare_runs.py`` read the ``network`` constants; ``network.ingest`` is the
canonical ingest mode (``udp``, ``daqiri-host``, ``gpudirect`` or ``replay``).
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import numpy as np

from ..nyq.mapping import PROTOCOL
from ..settings import GPU_INGEST_MODES

TRANSPORT = "FIX Simple Binary Encoding 1.0 / UDP · equity replay schema"
MESSAGE_COUNT_SCOPE = "SBE entries (one per observation); replay rate is source observations/s"
PAYLOAD_PATHS = {
    "gpudirect": "NIC -> GPU memory (GPUDirect) -> GPU gather -> GPU SBE decode",
    "replay": "pcap frames in GPU memory -> GPU gather -> GPU SBE decode",
}
PROFILE_KEYS = ("date", "events", "profile", "start_ns", "end_ns")
OBSERVATION_KEYS = ("events", "quotes", "trades", "invalid_quotes", "late_events", "completed_bars")


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _not_json(value: Any) -> Any:
    raise TypeError(f"/snapshot holds a {type(value).__name__}, which is not plain JSON")


def dumps(value: Any) -> str:
    """Strict compact JSON, as Starlette's ``JSONResponse`` renders it; a non-finite float
    (no JSON number carries it) is encoded as null rather than failing the snapshot."""
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=_not_json
        )
    except ValueError:
        return json.dumps(
            _finite(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            default=_not_json,
        )


def _finite(value: Any) -> Any:
    """``value`` with every non-finite float as None (containers copied)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


class DashboardModel:
    """Read side of the dashboard over an ``EquityContext`` (or any object with its
    attribute names: ``args``, ``settings``, ``directory``, ``feed``, ``results``, ``tap``,
    ``ingest_stats``, ``bars_stats``, ``risk_stats``, ``fullday``, ``status``,
    ``plan_summary``). ``args.ingest`` is the canonical ingest mode."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self._encoded: dict[str, tuple[Any, str]] = {}  # key -> (object, its JSON)

    def snapshot(self) -> dict:
        """``GET /snapshot`` (including ``engine``)."""
        result = self._snapshot()
        result["universe"] = list(result["universe"])
        return result

    def snapshot_json(self) -> bytes:
        """``snapshot()`` as UTF-8 JSON. The live result is replaced, never mutated, when a
        result is published, and the universe is the directory's tuple: each is encoded
        once and reused while the same object is current."""
        result = self._snapshot()
        blocks = [f'"{key}":{self._encode(key, result.pop(key))}' for key in ("live", "universe")]
        rest = dumps(result)
        return ("{" + ",".join([*blocks, rest[1:-1]] if len(rest) > 2 else blocks) + "}").encode()

    def _encode(self, key: str, value: Any) -> str:
        cached = self._encoded.get(key)
        if cached is not None and cached[0] is value:
            return cached[1]
        encoded = dumps(list(value) if isinstance(value, tuple) else value)
        self._encoded[key] = (value, encoded)
        return encoded

    def _snapshot(self) -> dict:
        """The snapshot, with the universe as the directory's own tuple."""
        c = self.ctx
        d = c.directory.current  # one lock-free read: every part below uses this generation
        view = c.results.view()
        errors = (c.status.error, view["error"], c.risk_stats.error, c.ingest_stats.error)
        result = dict(
            graph_status=c.status.graph_status,
            error=next((e for e in errors if e is not None), None),
            comparison_failure=view["comparison_failure"],
            settings=c.settings.as_params(),
            minimum_update_interval_seconds=c.args.interval,
            analytics_backend=c.args.analytics_backend,
            feed_control=c.feed.control(),
            sender_status=c.feed.sender,
            generation=d.generation if d is not None else 0,
            live=view["live"],
            comparison=view["comparison"],
            comparisons=view["comparisons"],
            history=view["history"],
            capture=c.tap.status(),
            profile={k: d.manifest[k] for k in PROFILE_KEYS} if d is not None else {},
            universe=d.symbols if d is not None else (),
        )
        if d is not None:  # observations exist from the first manifest on
            # One read: the bars lane replaces the counters per batch and drops them at a
            # generation change. Before this generation's first batch: zeros.
            counters = c.bars_stats.counters
            current = counters is not None and counters.generation == d.generation
            result["observations"] = {
                k: getattr(counters, k) if current else 0 for k in OBSERVATION_KEYS
            }
            prices = self._prices(d.generation, view["live"])
            if prices is not None:
                result["prices"] = prices
        if view["rolling_timing"] is not None:
            result["rolling_timing"] = view["rolling_timing"]
        result["network"] = self.network()
        f = c.fullday.status()
        result["fullday"] = dict(f, base=None) if f else None
        result["fullday_available"] = c.fullday.available()
        result["engine"] = self._engine()
        return result

    def _prices(self, generation: int, live: dict | None) -> list[dict] | None:
        """The 30 most volatile eligible instruments, ranked by annual volatility."""
        snap = self.ctx.bars_stats.symbol_snapshot
        if (
            not live
            or live["generation"] != generation
            or snap is None
            or snap.generation != generation
        ):
            return None
        indices = live["eligible_indices"]
        ranking = np.argsort(-np.asarray(live["annual_vol"]))[:30]
        return [
            dict(
                symbol=snap.symbols[indices[j]],
                mid=_number(snap.mid[indices[j]]),
                last_trade=_number(snap.trade[indices[j]]),
                observed_volume=_number(snap.volume[indices[j]]),
                annual_vol=live["annual_vol"][j],
                quote_age_s=float(snap.last_event_ns - snap.quote_time[indices[j]]) / 1e9,
            )
            for j in ranking
        ]

    def network(self) -> dict:
        """``/snapshot.network``: receive counters, the arrival-rate buckets and the
        transport constants. Before the source opens it is the starting dict."""
        c = self.ctx
        s, bars, mode = c.ingest_stats, c.bars_stats, c.args.ingest
        if not s.opened:
            return {"ready": False, "status": "starting", "ingest": mode}
        now = time.time_ns()
        buckets = s.feed_buckets(last=100)
        boundary = now // 100_000_000 * 100
        age = (now - s.last_packet_wall_ns) / 1e9 if s.last_packet_wall_ns else None
        return dict(
            ready=s.ready(),
            status="receiving"
            if age is not None and age < 2
            else "idle"
            if age is not None
            else "waiting",
            ingest=mode,
            transport=TRANSPORT,
            wire_protocol=PROTOCOL,
            message_count_scope=MESSAGE_COUNT_SCOPE,
            packets=s.packets,
            received_messages=s.received_messages,
            decoded_messages=s.messages,
            messages_per_second=sum(n for t, n in buckets if boundary - 1000 <= t < boundary),
            sequence_gaps=s.dropped,
            analytics_sequence_gaps=bars.sequence_gaps,
            ring_dropped=s.ring_dropped,
            ring_messages=s.ring_fill(),
            malformed_packets=s.malformed_packets + s.decode_malformed,
            out_of_order_packets=s.out_of_order_packets,
            stale_packets=s.stale_packets + bars.stale_packets,
            error=s.error,
            buckets=buckets,
            gpu_ingest=dict(
                bursts=s.bursts,
                non_udp_frames=s.non_udp_frames,
                drain_host_ms=s.drain_host_ns / 1e6,
                frame_fetch=s.frame_fetch,
                ring_peak=s.ring_peak(),
                payload_path=PAYLOAD_PATHS[mode],
            )
            if mode in GPU_INGEST_MODES
            else None,
        )

    def _engine(self) -> dict:
        """What the Holoscan 5 engine runs: lanes, edges and clocks from the compiled plan,
        termination, lane placement and the scheduling counters."""
        c = self.ctx
        plan = c.plan_summary or {}
        ingest, risk = c.ingest_stats, c.risk_stats
        replay = ingest.replay
        return dict(
            sdk=os.environ.get("HOLOSCAN_SDK_VERSION"),
            lanes=plan.get("lanes"),
            edges=plan.get("edges"),
            clocks=plan.get("clocks"),
            termination=c.status.termination,
            lane_cpus=dict(
                ingest=list(ingest.lane_cpus),
                bars=list(c.bars_stats.lane_cpus),
                risk=list(risk.lane_cpus),
            ),
            risk=dict(
                windows_superseded=risk.windows_superseded,
                gate_waits=risk.gate_waits,
                late_activations=risk.late_activations,
            ),
            source=dict(retries=ingest.retries, held=ingest.held),
            replay=dict(replay) if replay is not None else None,
        )

    def start_capture(self, payload: dict) -> dict:
        """``POST /capture``: needs a full return window. ValueError otherwise."""
        c = self.ctx
        seconds = payload.get("seconds", 120)
        if type(seconds) not in (int, float):  # float() would also take "5" or true
            raise ValueError("capture seconds must be a number")
        seconds = float(seconds)
        if c.tap.active:
            raise ValueError("finish the previous capture before starting another")
        live = c.results.live
        if not live or live["bars"] < c.settings.window_bars:
            raise ValueError("wait for a full return window before capturing")
        return c.tap.start(seconds, c.directory.generation, c.args.analytics_backend)

    def finish_capture(self) -> dict:
        """``POST /capture/finish``: the capture's result plus the run description
        (``compare_runs.py`` reads ``ingest``). ValueError if no capture was started."""
        c = self.ctx
        result = c.tap.finish()
        result["ingest"] = c.args.ingest
        result["settings"] = c.settings.as_params()
        result["wire_protocol"] = PROTOCOL
        result["minimum_update_interval_seconds"] = c.args.interval
        return result
