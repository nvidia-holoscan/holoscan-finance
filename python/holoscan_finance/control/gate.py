# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CPU cross-check gate: at most one validated GPU window in flight.

Risk claims the gate for a window it will validate and waits while a claim is open;
``ResultsStore.on_comparison`` releases it after the comparison is published. So every
validated GPU result has its CPU pair and comparisons never pile up (4.3 ``validate_next``
and the risk operator's ``ready`` callable). ``allowed`` suspends validation, e.g. during a
full-day burst (4.3 ``validation_allowed``).
"""

from __future__ import annotations

import threading
from collections.abc import Callable


class CrossCheckGate:
    in_flight: bool
    claims: int
    releases: int

    def __init__(self, *, allowed: Callable[[], bool] = lambda: True) -> None:
        self._allowed = allowed
        self._lock = threading.Lock()
        self.in_flight = False
        self.claims = self.releases = 0

    def ready(self) -> bool:
        """No comparison in flight."""
        return not self.in_flight

    def try_claim(self) -> bool:
        """Claim the gate for one validated window: True if allowed and not in flight."""
        # allowed() may take the full-day controller's lock: never call it under the gate's.
        if not self._allowed():
            return False
        with self._lock:
            if self.in_flight:
                return False
            self.in_flight = True
            self.claims += 1
            return True

    def release(self) -> None:
        """End the claim; a no-op when nothing is in flight."""
        with self._lock:
            if self.in_flight:
                self.in_flight = False
                self.releases += 1
