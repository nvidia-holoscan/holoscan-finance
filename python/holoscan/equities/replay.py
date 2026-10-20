# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory-mapped NYQ trade/quote replay over UDP with pause, speed and seek."""

import argparse
import json
import socket
import time
from urllib.request import Request, urlopen

import numpy as np

from holoscan.marketdata.wire import CONDITION, EVENT_DTYPE

from .data import open_tape
from .mitch import MAX_PACKET_EVENTS, PROTOCOL, TIMEZONE, day_start_ns, pack_events


def request(base, path, payload=None):
    req = Request(
        base.rstrip("/") + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=3) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="directory produced by equities.data")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=26400)
    p.add_argument("--control-url", default="http://127.0.0.1:8080")
    p.add_argument("--loops", type=int, default=0, help="0 repeats indefinitely")
    a = p.parse_args()
    manifest, rows = open_tape(a.data)
    sock = None
    used_ports = set()
    midnight_ns = day_start_ns(manifest["date"])
    target = (a.host, a.port)
    idx = sent = loops = 0
    applied_seek = -1
    generation = sequence = 0
    last_poll = anchor_wall = anchor_ts = 0
    previous_pacing = None

    def reset(position):
        nonlocal idx, generation, sequence, anchor_ts, anchor_wall, sock
        day_start = manifest["start_ns"] // (86400 * 10**9) * (86400 * 10**9)
        ts = np.clip(
            day_start + int(position * 10**9), manifest["start_ns"], manifest["end_ns"] - 1
        )
        idx = min(int(np.searchsorted(rows["timestamp_ns"], ts)), len(rows) - 1)
        generation = max(generation + 1, time.time_ns())
        sequence = 1
        if sock is not None:
            sock.close()
        for _ in range(100):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", 0))
            port = sock.getsockname()[1]
            if port not in used_ports:
                break
            sock.close()
        else:
            raise RuntimeError("no fresh UDP source port for MITCH replay generation")
        used_ports.add(port)
        accepted = request(
            a.control_url,
            "/manifest",
            {
                **manifest,
                "generation": generation,
                "wire_protocol": PROTOCOL,
                "market_timezone": TIMEZONE,
                "sender_port": port,
            },
        )
        if accepted["generation"] != generation:
            raise RuntimeError("receiver rejected the replay generation")
        anchor_ts, anchor_wall = int(rows["timestamp_ns"][idx]), time.perf_counter()

    print(
        f"MITCH NYQ replay: {len(rows):,} observations / {len(manifest['symbols']):,} RICs -> {target}",
        flush=True,
    )
    while a.loops == 0 or loops < a.loops:
        now = time.perf_counter()
        if now - last_poll >= 0.2 or not generation:
            last_poll = now
            try:
                updated = request(a.control_url, "/feedrate")
                pacing = (updated["speed"], updated["rate"], updated["paused"])
                if pacing != previous_pacing:
                    anchor_ts, anchor_wall = int(rows["timestamp_ns"][idx]), now
                    previous_pacing = pacing
                control = updated
                if applied_seek != control["seek_seq"]:
                    reset(control["seek"])
                    applied_seek = control["seek_seq"]
                request(
                    a.control_url,
                    "/feedstatus",
                    {
                        "seek_seq": applied_seek,
                        "generation": generation,
                        "position": float(rows["timestamp_ns"][idx] % (86400 * 10**9)) / 1e9,
                        "sent": sent,
                        "loops": loops,
                    },
                )
            except (OSError, ValueError) as exc:
                if not generation:
                    raise RuntimeError(
                        "receiver must accept the symbol directory before replay"
                    ) from exc
                print(f"Replay control temporarily unavailable: {exc}", flush=True)
                time.sleep(0.1)
                continue
        if control["paused"]:
            time.sleep(0.02)
            continue
        end = min(idx + MAX_PACKET_EVENTS, len(rows))
        if control["speed"] > 0:
            due = (
                anchor_wall
                + (int(rows["timestamp_ns"][end - 1]) - anchor_ts) / 1e9 / control["speed"]
            )
        else:
            due = anchor_wall
        delay = due - time.perf_counter()
        if delay > 0:
            time.sleep(min(0.01, delay))
            continue
        packet, wire_count = pack_events(rows[idx:end], sequence, midnight_ns)
        sock.sendto(packet, target)
        count = end - idx
        sequence += wire_count
        sent += count
        idx = end
        if control["speed"] == 0:
            anchor_wall = max(anchor_wall + count / control["rate"], time.perf_counter() - 0.1)
        if idx == len(rows):
            # Event-time watermark closes the final regular-session bar.
            marker = np.zeros(1, dtype=EVENT_DTYPE)
            marker["timestamp_ns"] = manifest["end_ns"]
            marker["kind"] = CONDITION
            packet, _ = pack_events(marker, sequence, midnight_ns)
            sock.sendto(packet, target)
            loops += 1
            if a.loops and loops >= a.loops:
                break
            reset(manifest["start_ns"] // 10**9 % 86400)
    sock.close()


if __name__ == "__main__":
    main()
