# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The run's state as ``/snapshot``, ``/healthz`` and the reports show it.

The runner (or the dashboard's watcher) mirrors the ``RunSession`` here, so HTTP handlers
never touch the session. A failed run reports its first error-severity diagnostic.
``ready()`` is what ``/healthz`` answers: the graph runs and its ingest front end receives
(``IngestStats.ready``: the socket bound or the NIC or replay drain running, no error), so
a sender started on it loses no packets. Once a finite input has ended (a replay's last
pass) the front end no longer receives, while the graph and the dashboard keep running:
``input_ended()`` tells that state from one that has not started. No Holoscan import: the
session's enums are read by name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class RunStatus:
    """The session's state for HTTP and reports; written by the runner or the watcher.
    ``receiving``: whether the ingest front end receives (None: the graph has none);
    ``input_ended``: whether its input ended cleanly (``IngestStats.eof``)."""

    graph_status: str  # "starting" | "running" | "stopped"
    error: str | None
    termination: str | None  # RunTermination name: RUNNING, COMPLETED, STOPPED, FAILED, ...

    def __init__(
        self,
        receiving: Callable[[], bool] | None = None,
        input_ended: Callable[[], bool] | None = None,
    ) -> None:
        self.graph_status = "starting"
        self.error = self.termination = None
        self._receiving, self._input_ended = receiving, input_ended

    def ready(self) -> bool:
        """The graph runs, without an error, and its front end receives."""
        if self.graph_status != "running" or self.error is not None:
            return False
        return self._receiving is None or self._receiving()

    def input_ended(self) -> bool:
        """The front end's input ended and everything it received was emitted (a finite
        replay after its last pass, or a source without ``keep_alive`` after its receive
        timeout): ``ready()`` is False from then on, though the graph may still run."""
        return self._input_ended is not None and bool(self._input_ended())

    def from_session(self, session: Any) -> None:
        """Mirror ``session`` (an ``hs.RunSession``); safe while it is still running."""
        self.termination = session.termination.name
        if not session.is_running:
            self.graph_status = "stopped"
        if self.termination == "FAILED" and self.error is None:
            self.error = _first_error(session.diagnostics) or "run failed"


def _first_error(diagnostics: Any) -> str | None:
    """``"<operator>: <last line of the error>"`` of the first error-severity diagnostic.
    The full traceback stays in ``detail.error.message`` (and the SDK's stderr report)."""
    for d in diagnostics:
        if d.severity.name != "ERROR":
            continue
        detail = d.detail
        error = getattr(detail, "error", None)
        lines = (getattr(error, "message", None) or "").strip().splitlines()
        text = lines[-1] if lines else d.token
        where = getattr(detail, "operator_path", None)
        return f"{where}: {text}" if where else text
    return None
