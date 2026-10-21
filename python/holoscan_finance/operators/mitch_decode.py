# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``MitchDecodeOp``: the Python reference MITCH decoder of the host front ends.

Ports: ``packets`` (in, ``PacketBatch``) -> ``events`` (out, ``list[ReplayBatch]``, one
batch per accepted packet). Edges: ``lossless(16)`` from the source and ``lossless(16)``
into ``QuoteBarsOp(backend="cpu")``, all in the ``ingest`` lane (no hop per packet). The
directory is read once per batch, immediately before decoding, as 4.3 ``EquityState.decode``
did, so a ``POST /manifest`` reaches the next batch; packets from another sender port are
stale. Processor: ``nyq.host_decoder.HostMitchDecoder`` (the 0.28M msgs/s host baseline).
"""

from __future__ import annotations

import holoscan as hs

from ..nyq.host_decoder import HostMitchDecoder
from ..ref import Ref
from ..telemetry.stats import IngestStats
from ._common import ref_obj


class MitchDecodeOp(hs.Operator):
    """Decode every ``PacketBatch`` under the current feed directory.

    Writes only ``messages``, ``stale_packets``, ``decode_malformed``, ``event_batches`` and
    ``packet_batches_decoded`` of the ``IngestStats`` it shares with its source.
    """

    packets = hs.Input("packets", hs.pyobject)
    events = hs.Output("events", hs.pyobject)

    directory = hs.Param(
        Ref(), description="Ref(Knob[FeedDirectory]); empty: every packet is stale"
    )
    stats = hs.Param(Ref(), description="Ref(IngestStats), normally the source's; private if empty")
    tap = hs.Param(Ref(), description="Ref(CaptureTap); no capture recording if empty")

    def __init__(self) -> None:
        self._decoder = self._knob = self._stats = self._tap = None

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnEach(self.packets))

    def start(self) -> None:
        self._decoder = HostMitchDecoder()
        self._knob = self.directory.obj
        self._stats = ref_obj(self.stats, IngestStats)
        self._tap = self.tap.obj

    def compute(self, ctx: hs.ExecutionContext) -> None:
        batch = self.packets.receive_data()
        out, n = self._decoder.decode(batch, self._knob.get() if self._knob is not None else None)
        st = self._stats
        st.messages += n.messages
        st.stale_packets += n.stale
        st.decode_malformed += n.malformed
        if self._tap is not None and self._tap.active:
            self._tap.record_host_batches(out)
        if out:
            self.events.emit(out)
            st.event_batches += 1
        # Last: a finite run's drained check reads "every packet batch decoded" as "every
        # event batch counted", so this must not run ahead of event_batches.
        st.packet_batches_decoded += 1
