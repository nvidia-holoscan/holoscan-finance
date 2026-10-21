# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ReplayDirectory and ReplayGenerations: manifest rules, listener order, lock release."""

from __future__ import annotations

import threading
import time

import pytest

from holoscan_finance.control.directory import ReplayDirectory, ReplayGenerations
from holoscan_finance.nyq.mapping import PROTOCOL, TIMEZONE, day_start_ns
from holoscan_finance.ref import Knob

MIDNIGHT = day_start_ns("2025-04-01")


def manifest(generation=1, port=50001, **extra):
    return dict(
        format="nyq-equity-replay-v1",
        wire_protocol=PROTOCOL,
        market_timezone=TIMEZONE,
        symbols=["LONG_RIC", "OTHER"],
        generation=generation,
        sender_port=port,
        date="2025-04-01",
        start_ns=MIDNIGHT + 34200 * 10**9,
        end_ns=MIDNIGHT + 57600 * 10**9,
        events=100,
        profile=[[34200, 10]],
        **extra,
    )


def test_generations_are_monotonic_and_idempotent():
    knob = Knob()
    d = ReplayDirectory(knob)
    assert d.current is None and d.generation == 0
    assert d.set_manifest(manifest(5)) == {"generation": 5}
    assert d.generation == 5 and d.current.sender_port == 50001 and knob.version == 1
    # An equal or older generation is a no-op that reports the current one.
    assert d.set_manifest(manifest(5, port=50002)) == {"generation": 5}
    assert d.set_manifest(manifest(3, port=50003)) == {"generation": 5}
    assert knob.version == 1 and d.used_sender_ports == {50001}


def test_seek_takes_a_new_port_and_rejects_the_old_one():
    # A seek registers a new generation from a new sender port; packets still queued from
    # the old port are stale because the directory names the new one.
    d = ReplayDirectory(Knob())
    d.set_manifest(manifest(1, 50001))
    d.set_manifest(manifest(2, 50002))
    assert d.current.generation == 2 and d.current.sender_port == 50002
    with pytest.raises(ValueError, match="port must be new"):
        d.set_manifest(manifest(3, 50001))
    assert d.generation == 2


@pytest.mark.parametrize(
    "bad",
    [
        dict(format="other"),
        dict(sender_port=0),
        dict(symbols=["A", "A"]),
        dict(generation=0),
        dict(date="not-a-date"),
        dict(start_ns=5, end_ns=5),
    ],
)
def test_invalid_manifests_never_replace_the_generation(bad):
    d = ReplayDirectory(Knob())
    d.set_manifest(manifest(1))
    with pytest.raises(ValueError):
        d.set_manifest({**manifest(2, 50002), **bad})
    assert d.generation == 1 and 50002 not in d.used_sender_ports


def test_listeners_run_in_registration_order_with_the_new_directory():
    d = ReplayDirectory(Knob())
    calls = []
    d.add_listener(lambda fd: calls.append(("results", fd.generation)))
    d.add_listener(lambda fd: calls.append(("tap", fd.generation, d.generation)))
    d.set_manifest(manifest(1))
    d.set_manifest(manifest(1, 50002))  # no-op: no listener runs
    d.set_manifest(manifest(2, 50002))
    assert calls == [("results", 1), ("tap", 1, 1), ("results", 2), ("tap", 2, 2)]


def test_listeners_run_after_the_directory_lock_is_released():
    d = ReplayDirectory(Knob())
    seen = []

    def listener(fd):
        # Another thread (a data-plane reader or a second writer's validation) must be able
        # to take the directory lock and read the directory while a listener runs.
        def other():
            with d.lock:
                seen.append(d.current.generation)

        t = threading.Thread(target=other)
        t.start()
        t.join(timeout=5)
        seen.append(not t.is_alive())

    d.add_listener(listener)
    d.set_manifest(manifest(7))
    assert seen == [7, True]


def test_listener_runs_keep_publication_order_across_writers():
    d = ReplayDirectory(Knob())
    seen = []

    def slow_on_odd(fd):
        # A slow listener run must hold back the next generation's listeners, or a newer
        # generation's reset could be overwritten by an older one's.
        if fd.generation % 2:
            time.sleep(0.001)
        seen.append(fd.generation)

    d.add_listener(slow_on_odd)
    ports = iter(range(20000, 30000))
    lock = threading.Lock()

    def writer(offset):
        for k in range(100):
            with lock:
                port = next(ports)
            d.set_manifest(manifest(2 * k + offset, port))

    threads = [threading.Thread(target=writer, args=(o,)) for o in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == sorted(seen) and len(set(seen)) == len(seen)
    assert seen[-1] == d.generation == 200


def test_a_failing_listener_releases_the_listener_lock():
    d = ReplayDirectory(Knob())

    def boom(fd):
        raise RuntimeError("listener failed")

    d.add_listener(boom)
    with pytest.raises(RuntimeError):
        d.set_manifest(manifest(1))
    assert d.generation == 1  # published before the listeners ran
    with pytest.raises(RuntimeError):
        d.set_manifest(manifest(2, 50002))  # would hang if the listener lock leaked


def test_replay_generations_register_each_later_pass_from_a_fresh_port():
    d = ReplayDirectory(Knob())
    base = {k: v for k, v in manifest().items() if k not in ("sender_port", "generation")}
    d.set_manifest({**base, "sender_port": 40000, "generation": 10})
    seen = []
    d.add_listener(lambda fd: seen.append((fd.generation, fd.sender_port)))
    loops = ReplayGenerations(d, base, first_port=40000)
    assert loops(0) is None and not seen  # pass 0 is the context's registration
    d.used_sender_ports.add(40001)  # e.g. taken by a manual POST /manifest
    assert loops(1) == 40002
    assert loops(2) == 40003
    assert [p for _, p in seen] == [40002, 40003]
    assert 10 < seen[0][0] < seen[1][0] == d.generation
    assert d.current.symbols == ("LONG_RIC", "OTHER") and d.current.sender_port == 40003


def test_replay_generations_wrap_within_unprivileged_ports():
    d = ReplayDirectory(Knob())
    base = {k: v for k, v in manifest().items() if k not in ("sender_port", "generation")}
    loops = ReplayGenerations(d, base, first_port=65535)
    assert loops(1) == 1024
    d.used_sender_ports.update(range(1024, 1030))
    assert loops(2) == 1030


def test_replay_generations_stay_above_a_manual_generation():
    d = ReplayDirectory(Knob())
    base = {k: v for k, v in manifest().items() if k not in ("sender_port", "generation")}
    huge = 2**62
    d.set_manifest({**base, "sender_port": 30000, "generation": huge})
    port = ReplayGenerations(d, base, first_port=30000)(1)
    assert port == 30001 and d.generation == huge + 1 and d.current.sender_port == port
