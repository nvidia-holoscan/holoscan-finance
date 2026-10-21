# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The run's state as ``/snapshot``, ``/healthz`` and the reports show it.

The runner (or the dashboard's watcher) mirrors the ``RunSession`` here, so HTTP handlers
never touch the session. A failed run reports its first error-severity diagnostic. No
Holoscan import: the session's enums are read by name.
"""

from __future__ import annotations

from typing import Any


class RunStatus:
    """The session's state for HTTP and reports; written by the runner or the watcher."""

    graph_status: str  # "starting" | "running" | "stopped"
    error: str | None
    termination: str | None  # RunTermination name: RUNNING, COMPLETED, STOPPED, FAILED, ...

    def __init__(self) -> None:
        self.graph_status = "starting"
        self.error = self.termination = None

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
