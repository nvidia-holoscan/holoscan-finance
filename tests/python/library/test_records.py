# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decoded observation batches: coalescing keeps every packet's receive stamp, and sequence
counts are wire messages when the decoder knows them; a device bar window stacks on its own
device."""

from __future__ import annotations

import numpy as np

from holoscan_finance.marketdata.records import EVENT_DTYPE, ReplayBatch, combine_batches
from holoscan_finance.payloads import BarWindow


def batch(n, recv_ns, sequence, wire_count=None, generation=1):
    rows = np.zeros(n, EVENT_DTYPE)
    rows["timestamp_ns"] = recv_ns + np.arange(n)
    return ReplayBatch(rows, generation, recv_ns, sequence, wire_count=wire_count)


def test_sequence_count_is_wire_messages_when_known():
    assert batch(3, 10, 1).sequence_count == 3
    assert batch(3, 10, 1, wire_count=5).sequence_count == 5
    assert batch(0, 10, 1, wire_count=1).sequence_count == 1


def test_combine_keeps_packet_offsets_and_receive_stamps():
    parts = [batch(2, 100, 1), batch(0, 200, 3), batch(3, 300, 4)]
    combined = combine_batches(parts)
    assert len(combined.events) == 5
    assert (combined.generation, combined.sequence, combined.recv_wall_ns) == (1, 1, 300)
    assert combined.packet_offsets.tolist() == [0, 2, 2]
    assert combined.packet_recv_ns.tolist() == [100, 200, 300]
    assert combined.packet_recv_ns.dtype == np.int64
    np.testing.assert_array_equal(
        combined.events["timestamp_ns"], np.r_[parts[0].events, parts[2].events]["timestamp_ns"]
    )


def test_a_single_batch_passes_through():
    one = batch(2, 100, 7)
    assert combine_batches([one]) is one


def test_a_device_window_stacks_on_its_parts_device(monkeypatch):
    """``BarWindow.rows`` of device parts stacks under the parts' device, not whichever
    device is current in the consumer's thread (a capture records on the publish lane)."""
    import sys
    import types

    current = ["device 0"]

    class Device:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            self.previous, current[0] = current[0], self.name

        def __exit__(self, *exc):
            current[0] = self.previous

    class Part(np.ndarray):
        device = Device("device 1")

    stacked_on = []

    def stack(parts):
        stacked_on.append(current[0])
        return np.stack(parts)

    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(stack=stack))
    parts = tuple(np.zeros(3).view(Part) for _ in range(2))
    window = BarWindow(parts, np.arange(2), 1, 2, 0, device=True)
    assert window.rows.shape == (2, 3) and stacked_on == ["device 1"]
    assert current == ["device 0"]
