# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Receive accounting of every source: one implementation of the sequence rules.

Units are counted per sender session (source address and market-data group, so a new replay
generation's sender is never read as the current one): a sequence above the expected one
adds the gap to ``dropped`` (sequence gaps, in messages), one below it is out of order, and
the expectation never moves back. The host receivers call ``observe_packet`` per datagram;
the GPU path calls ``observe_burst`` on the 7-column receive header of a burst, vectorized
for the normal single-session contiguous burst and per packet otherwise (``_account_slow``),
with identical counters. Ring admission (``ring_dropped``) is left to the owner of each ring.
"""

from __future__ import annotations

import numpy as np

from ..marketdata.mitch_codec import inspect_unit
from ..telemetry.stats import IngestStats


def mitch_packet_header(data: bytes, peer: tuple[str, int] | None) -> tuple[tuple, int, int]:
    """``(session, sequence, message count)`` of one MITCH unit; ValueError if malformed.

    Sequences count wire messages, Time messages included.
    """
    group, sequence, count = inspect_unit(data)
    return (peer, group), sequence, count


class ReceiveAccounting:
    """Sequence, packet, message and arrival-bucket counters of one source's ``IngestStats``.

    One thread observes (the receiver's drain thread, or the GPU source's lane)."""

    def __init__(self, stats: IngestStats) -> None:
        self.stats = stats
        self._session: tuple | None = None
        self._expected: int | None = None  # next sequence of the current session

    def observe_packet(self, session: tuple, seq: int, count: int, recv_ns: int) -> None:
        """Account one framed unit that arrived at ``recv_ns`` (wall clock)."""
        st = self.stats
        self._sequence(session, seq, count)
        st.last_packet_wall_ns = recv_ns
        st.packets += 1
        st.received_messages += count
        with st.lock:
            st.add_bucket(recv_ns, count)

    def observe_burst(self, header_t: np.ndarray, recv_ns: int) -> np.ndarray:
        """Account one burst from its receive header, ``7 x n`` (payload length, source port,
        source IP, framing status, message count, sequence, group); returns the rows that
        are UDP and passed the framing check, the candidates for decode."""
        length, sport, sip, status, count, seq, group = header_t
        st = self.stats
        st.bursts += 1
        udp = length >= 0
        st.non_udp_frames += int((~udp).sum())  # strip_udp_datagram -> None: never framed
        st.malformed_packets += int((udp & (status == 1)).sum())
        cand = np.flatnonzero(udp & (status != 1))
        if not len(cand):
            return cand
        seq_c, cnt_c = seq[cand], count[cand]
        sess = np.stack([sip[cand], sport[cand], group[cand]], axis=1)
        single = len(cand) == 1 or bool((sess == sess[0]).all())
        contiguous = len(cand) == 1 or bool(np.array_equal(seq_c[1:], seq_c[:-1] + cnt_c[:-1]))
        first = int(seq_c[0])
        session0 = tuple(int(x) for x in sess[0])
        expected = self._expected if session0 == self._session else None
        if single and contiguous and (expected is None or first >= expected):
            # The normal burst: one sender, no reordering inside, nothing older than expected.
            self._session = session0
            if expected is not None and first > expected:
                st.dropped += first - expected
                st.stall_event("gap", 0, messages=first - expected)
            self._expected = int(seq_c[-1] + cnt_c[-1])
        else:
            for s, q, c in zip(map(tuple, sess.tolist()), seq_c.tolist(), cnt_c.tolist()):
                self._sequence(s, q, c)
        st.packets += len(cand)
        msgs = int(cnt_c.sum())
        st.received_messages += msgs
        st.last_packet_wall_ns = recv_ns
        with st.lock:
            st.add_bucket(recv_ns, msgs)
        return cand

    def _sequence(self, session: tuple, seq: int, count: int) -> None:
        """The per-packet rule."""
        if session != self._session:
            self._session, self._expected = session, None
        expected = self._expected
        if expected is not None and seq > expected:
            self.stats.dropped += seq - expected
        if expected is not None and seq < expected:
            self.stats.out_of_order_packets += 1
        self._expected = max(expected or seq, seq + count)


__all__ = ["ReceiveAccounting", "mitch_packet_header"]
