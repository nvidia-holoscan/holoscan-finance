# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Ref`` and ``Knob`` (the live-object handles operators share), the bar settings and the
ingest mode names."""

from __future__ import annotations

import copy
import dataclasses
import threading

import pytest

import holoscan_finance
from holoscan_finance.ref import Knob, Ref
from holoscan_finance.settings import (
    DAQIRI_INGEST_MODES,
    GPU_INGEST_MODES,
    INGEST_ALIASES,
    INGEST_MODES,
    LEGACY_INGEST_TOKEN,
    BarSettings,
    IngestDefaults,
    canonical_ingest,
)


def test_ref_copies_are_the_same_handle():
    target = {"live": 1}
    ref = Ref(target)
    nested = {"params": [ref]}
    assert copy.deepcopy(ref) is ref and copy.copy(ref) is ref
    assert copy.deepcopy(nested)["params"][0].obj is target
    assert bool(ref) and not Ref() and not Ref(None)
    assert repr(ref) == "Ref(dict)" and repr(Ref()) == "Ref(None)"
    assert bool(Ref(0)) and bool(Ref([]))  # set, though the object itself is falsy
    with pytest.raises(AttributeError):
        ref.other = 1  # slots: one field only


def test_knob_publishes_versioned_values():
    knob = Knob()
    assert knob.get() is None and knob.version == 0
    assert knob.set("a") == 1 and knob.get() == "a"
    assert Knob("b").get() == "b" and Knob("b").version == 0

    def write():
        for i in range(1000):
            knob.set(i)

    threads = [threading.Thread(target=write) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert knob.version == 4001  # every set got its own version


def test_package_root_resolves_names_lazily():
    assert holoscan_finance.Ref is Ref and holoscan_finance.Knob is Knob
    assert "Ref" in dir(holoscan_finance) and "QuoteBarsOp" in holoscan_finance.__all__
    assert holoscan_finance.__version__ == "0.2.0"
    with pytest.raises(AttributeError):
        holoscan_finance.NotAName


def test_bar_settings_are_the_4_3_defaults_and_rules():
    s = BarSettings()
    assert s.as_params() == dict(
        bar_seconds=5, window_seconds=300, stale_seconds=30, min_bars=20, max_spread_bps=100.0
    )
    assert s.window_bars == 60
    assert BarSettings(bar_seconds=10, window_seconds=600).window_bars == 60
    for bad in (
        dict(bar_seconds=0),
        dict(window_seconds=301),
        dict(stale_seconds=4),
        dict(min_bars=61),
        dict(max_spread_bps=-1.0),
    ):
        with pytest.raises(ValueError):
            BarSettings(**bad)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.bar_seconds = 1


def test_ingest_defaults_are_the_measured_gpudirect_settings():
    d = IngestDefaults()
    assert (d.gpu_batch_messages, d.gpu_ring_size, d.gpu_num_bufs) == (131072, 1 << 20, 114688)
    assert (d.host_batch_messages, d.host_ring_size) == (16384, 65536)
    assert (d.interval_s, d.burst_packets, d.rx_timeout_us) == (0.05, 4096, 20000)


def test_ingest_names_and_legacy_tokens():
    assert INGEST_MODES == ("udp", "daqiri-host", "gpudirect", "replay")
    for mode in INGEST_MODES:
        assert canonical_ingest(mode) == (mode, False)
    for alias, mode in INGEST_ALIASES.items():
        assert canonical_ingest(alias) == (mode, True)
        assert LEGACY_INGEST_TOKEN[mode] == alias  # /snapshot keeps the 4.3 token
    assert LEGACY_INGEST_TOKEN["replay"] == "replay"
    assert set(GPU_INGEST_MODES) | set(DAQIRI_INGEST_MODES) == {
        "daqiri-host",
        "gpudirect",
        "replay",
    }
    with pytest.raises(ValueError, match="unknown ingest mode"):
        canonical_ingest("rdma")
