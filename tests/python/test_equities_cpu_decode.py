# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The compiled CPU MITCH decoder must be bit-identical to the Python reference.

Same checks as the GPU decoder: golden bytes, randomized streams, mutated packets
exercising every rejection rule, sign-magnitude prices above 2**53, and every
thread count. Needs numba, not a GPU.
"""

import struct

import numpy as np
import pytest

pytest.importorskip("numba")

from holoscan.equities import gpu_validate as gv  # noqa: E402
from holoscan.equities import mitch as replay  # noqa: E402
from holoscan.equities.cpu_mitch import decode_packets  # noqa: E402
from holoscan.marketdata.gpu_mitch import SLOT_BYTES  # noqa: E402

MIDNIGHT = replay.day_start_ns("2025-04-01")
UNIVERSE = 50
from test_mitch import GOLDEN  # noqa: E402


def _slots(payloads):
    lens = np.fromiter((len(p) for p in payloads), np.int64, count=len(payloads))
    slots = np.zeros((len(payloads), SLOT_BYTES), np.uint8)
    for i, p in enumerate(payloads):
        slots[i, : len(p)] = np.frombuffer(p, np.uint8)
    return slots, lens


def _check(payloads, threads=None):
    import numba

    gv._CTX.update(midnight_ns=MIDNIGHT, universe=UNIVERSE)
    host = gv._host_decode(payloads)
    if threads:
        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    slots, lens = _slots(payloads)
    report = gv.compare(host, decode_packets(slots, lens, midnight_ns=MIDNIGHT, universe=UNIVERSE))
    bad = {k: v for k, v in report.items() if k.endswith("mismatch") and v}
    assert not bad, report
    assert report["events_host"] == report["events_gpu"]
    return report


def test_golden_unit():
    assert _check([GOLDEN])["events_gpu"] == 1


@pytest.mark.parametrize("threads", [1, 4])
def test_random_streams_and_mutations(threads):
    rng = np.random.default_rng(20261002)
    payloads = gv.synthetic_stream(rng, 400, UNIVERSE, MIDNIGHT)
    assert _check(payloads, threads)["status_counts_host"] == {0: len(payloads)}
    report = _check(gv.fuzz(payloads, 6000, seed=13), threads)
    assert set(report["status_counts_host"]) == {0, 1, 2}


def test_huge_prices_patched_exactly():
    unit = bytearray(GOLDEN)
    for magnitude in (2**53 + 1, 2**60 + 12345, 2**63 - 1, 123456789012345678):
        struct.pack_into("<Q", unit, 24, magnitude | (1 << 63))
        struct.pack_into("<Q", unit, 36, magnitude)
        _check([bytes(unit)])
