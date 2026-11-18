# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ``daqiri`` double that delivers raw Ethernet frames through the real burst-polling API.

The real engine needs a ConnectX NIC, hugepages and root. ``make_fake_daqiri(bursts)``
returns a module whose ``get_rx_burst`` hands out the given bursts (lists of frames) in
order and then reports idle; ``push`` adds more while a drain runs. Host bindings serve
frames as bytes (``get_packet_bytes``); with ``device=True`` every burst is uploaded to
the GPU first and ``get_packet_ptr``/``get_packet_length`` return device addresses, as
GPUDirect leaves frames. ``daqiri_init``, ``shutdown`` and the loader stand-ins append to
``calls``, so tests can check their order; ``pin`` makes ``daqiri_init`` re-pin its
calling thread, as DPDK's EAL does. Replace an attribute to inject a failure.
"""

from __future__ import annotations

import os
import threading
import types
from collections.abc import Iterable

import numpy as np


class Status:
    SUCCESS = "SUCCESS"
    NULL_PTR = "NULL_PTR"
    NOT_READY = "NOT_READY"


class _DeviceBurst:
    """Frames of one burst in device memory (kept alive with the burst)."""

    def __init__(self, frames: list[bytes], gpu_device: int) -> None:
        import cupy as cp

        lengths = np.array([len(f) for f in frames], np.uint64)
        offsets = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.uint64)
        with cp.cuda.Device(gpu_device):
            self.data = cp.asarray(np.frombuffer(b"".join(frames), np.uint8))
            cp.cuda.Device(gpu_device).synchronize()  # drains gather on other streams
        self.ptrs = np.uint64(self.data.data.ptr) + offsets
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.lengths)


def make_fake_daqiri(
    bursts: Iterable[list[bytes]] = (),
    *,
    device: bool = False,
    gpu_device: int = 0,
    pin: Iterable[int] | None = None,
    calls: list[str] | None = None,
) -> types.ModuleType:
    """A fresh fake ``daqiri`` module; install it with ``monkeypatch.setitem(sys.modules,
    "daqiri", fake)``."""
    mod = types.ModuleType("daqiri")
    mod.__version__ = "0.0+fake"
    mod.Status = Status
    mod.calls = [] if calls is None else calls
    mod.configs, mod.freed = [], []
    mod.shutdowns = 0
    queue: list = []
    lock = threading.Lock()

    def push(frames: list[bytes]) -> None:
        burst = _DeviceBurst(frames, gpu_device) if device else list(frames)
        with lock:
            queue.append(burst)

    def daqiri_init(config):
        mod.calls.append("daqiri_init")
        mod.configs.append(config)
        if pin is not None:
            os.sched_setaffinity(0, set(pin))
        return Status.SUCCESS

    def get_rx_burst(port, queue_id):
        with lock:
            if queue:
                return Status.SUCCESS, queue.pop(0)
        return Status.NULL_PTR, None

    def shutdown():
        mod.calls.append("shutdown")
        mod.shutdowns += 1

    mod.push = push
    mod.daqiri_init = daqiri_init
    mod.get_port_id = lambda name: 0
    mod.get_rx_burst = get_rx_burst
    mod.get_num_packets = len
    mod.free_all_packets_and_burst_rx = mod.freed.append
    mod.print_stats = lambda: None
    mod.shutdown = shutdown
    if device:
        mod.get_packet_ptr = lambda burst, i: int(burst.ptrs[i])
        mod.get_packet_length = lambda burst, i: int(burst.lengths[i])
    else:
        mod.get_packet_bytes = lambda burst, i: (Status.SUCCESS, burst[i])
    for frames in bursts:
        push(frames)
    return mod


__all__ = ["Status", "make_fake_daqiri"]
