# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The compiled CPU SBE decoder must be bit-identical to the Python reference.

Same checks as the GPU decoder: golden bytes, randomized streams, mutated packets
exercising every rejection rule, the PriceNULL4 value edges, sequences at the int64 edge,
schema extensions and every thread count. Needs numba>=0.68, not a GPU; skipped without it.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
from library.test_sbe_codec import GOLDEN, extended

from holoscan_finance.marketdata import sbe_codec as sbe
from holoscan_finance.marketdata.gpu_sbe import SLOT_BYTES
from holoscan_finance.nyq.mapping import day_start_ns
from holoscan_finance.nyq.synthetic import fuzz, synthetic_stream
from holoscan_finance.validation.compare import compare_decoded, host_decode

cpu_sbe = pytest.importorskip("holoscan_finance.tools.cpu_sbe")  # needs numba>=0.68

MIDNIGHT = day_start_ns("2025-04-01")
UNIVERSE = 50


def _slots(payloads):
    lens = np.fromiter((len(p) for p in payloads), np.int64, count=len(payloads))
    slots = np.zeros((len(payloads), SLOT_BYTES), np.uint8)
    for i, p in enumerate(payloads):
        slots[i, : len(p)] = np.frombuffer(p, np.uint8)
    return slots, lens


def _check(payloads, threads=None):
    import numba

    host = host_decode(payloads, MIDNIGHT, UNIVERSE, workers=1)
    if threads:
        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    slots, lens = _slots(payloads)
    decoded = cpu_sbe.decode_packets(slots, lens, midnight_ns=MIDNIGHT, universe=UNIVERSE)
    report = compare_decoded(host, decoded)
    bad = {k: v for k, v in report.items() if k.endswith("mismatch") and v}
    assert not bad, report
    assert report["events_host"] == report["events_gpu"]
    return report


def test_golden_unit():
    assert _check([GOLDEN])["events_gpu"] == 1


@pytest.mark.parametrize("threads", [1, 4])
def test_random_streams_and_mutations(threads):
    rng = np.random.default_rng(20261002)
    payloads = synthetic_stream(rng, 400, UNIVERSE, MIDNIGHT)
    assert _check(payloads, threads)["status_counts_host"] == {0: len(payloads)}
    report = _check(fuzz(payloads, 6000, seed=13), threads)
    assert set(report["status_counts_host"]) == {0, 1, 2}


def test_value_edges_sequence_edges_and_extensions():
    unit = bytearray(GOLDEN)
    for bid, ask in ((2**63 - 1, sbe.PRICE_NULL + 1), (2**53 + 1, -(2**53) - 1), (1, -1)):
        struct.pack_into("<q", unit, 24 + 8, bid)
        struct.pack_into("<q", unit, 24 + 16, ask)
        report = _check([bytes(unit)])
        assert report["events_gpu"] == 1
        assert report["max_price_magnitude"] == max(abs(float(bid)), abs(float(ask))) / 1e4
    for sequence in (1, 2**63 - 3, 2**63 - 2, 2**63 - 1, 2**63, 2**64 - 1):
        struct.pack_into("<Q", unit, 8, sequence)
        _check([bytes(unit)])
    report = _check([extended(), extended(8, 0), extended(0, 16), extended(4, 0), extended(0, 4)])
    assert report["status_counts_host"] == {0: 3, 1: 2}
