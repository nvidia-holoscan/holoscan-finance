# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protocol fakes for the analytics consumers: a results store and a cross-check gate.

``ResultsStoreFake`` has the two ``CallbackSinkOp`` callbacks of ``control.ResultsStore``
(``on_risk``, ``on_comparison``) and its call counters, and releases the gate after
recording a comparison, as the real store does after publishing it. ``GateFake`` is a
``CrossCheckGate`` that logs every call, so tests can assert who claims and who releases.
Both are thread-safe: their callers run on runtime lanes.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from holoscan_finance.control.gate import CrossCheckGate


class GateFake:
    """``CrossCheckGate`` protocol; ``calls`` logs ``ready``/``try_claim``/``release`` in order."""

    def __init__(self, *, allowed: Callable[[], bool] = lambda: True) -> None:
        self.allowed = allowed
        self.in_flight = False
        self.claims = self.releases = 0
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def ready(self) -> bool:
        with self._lock:
            self.calls.append("ready")
            return not self.in_flight

    def try_claim(self) -> bool:
        with self._lock:
            self.calls.append("try_claim")
            if not self.allowed() or self.in_flight:
                return False
            self.in_flight = True
            self.claims += 1
            return True

    def release(self) -> None:
        with self._lock:
            self.calls.append("release")
            if self.in_flight:
                self.in_flight = False
                self.releases += 1


class ResultsStoreFake:
    """Records every ``RiskUpdate`` and ``RiskComparison`` it is handed, in order."""

    def __init__(self, gate: GateFake | CrossCheckGate | None = None) -> None:
        self.gate = gate
        self.risk: list = []
        self.comparisons: list = []
        self._lock = threading.Lock()

    @property
    def received_risk(self) -> int:
        return len(self.risk)

    @property
    def received_comparisons(self) -> int:
        return len(self.comparisons)

    def on_risk(self, update: object) -> None:
        with self._lock:
            self.risk.append(update)

    def on_comparison(self, comparison: object) -> None:
        with self._lock:
            self.comparisons.append(comparison)
        if self.gate is not None:  # after publication: the next validated window may start
            self.gate.release()
