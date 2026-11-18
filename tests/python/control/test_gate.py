# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CrossCheckGate: one validated window in flight, released after its comparison."""

from __future__ import annotations

import threading

from holoscan_finance.control.gate import CrossCheckGate


def test_claim_blocks_until_released():
    gate = CrossCheckGate()
    assert gate.ready() and gate.try_claim()
    assert gate.in_flight and not gate.ready()
    assert not gate.try_claim()  # one comparison at a time
    gate.release()
    assert gate.ready() and gate.try_claim()
    assert (gate.claims, gate.releases) == (2, 1)


def test_release_without_claim_is_a_no_op():
    gate = CrossCheckGate()
    gate.release()
    gate.try_claim()
    gate.release()
    gate.release()
    assert (gate.claims, gate.releases) == (1, 1) and gate.ready()


def test_allowed_suspends_claims_without_blocking_risk():
    allowed = [False]
    gate = CrossCheckGate(allowed=lambda: allowed[0])
    assert not gate.try_claim()
    assert gate.ready()  # a suspended gate never makes risk wait
    allowed[0] = True
    assert gate.try_claim() and gate.claims == 1


def test_concurrent_claims_admit_exactly_one():
    gate = CrossCheckGate()
    barrier = threading.Barrier(8)
    won = []

    def claim():
        barrier.wait()
        won.append(gate.try_claim())

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(won) == 1 and gate.claims == 1
