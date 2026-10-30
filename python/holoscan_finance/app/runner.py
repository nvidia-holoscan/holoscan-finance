# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run plumbing: process set-up, compile against the app's rules, headless run, report.

``prepare_process`` applies the process settings before any other thread exists: the 1 ms
GIL switch interval and the main thread masked to ``helper``, plus a sweep of earlier
threads (``keep_off_cores``), so the runtime's threads, uvicorn, the stall heartbeat, CuPy and spawned workers all inherit that
mask. ``helper`` is ``app`` (off the DPDK lcores in DAQIRI modes) unless a busy-polled
ingest set is carved out, which then keeps every other thread off the polled CPUs
(``cpu_sets`` is the same computation with no side effects). ``compile_plan`` stops before
anything starts if the plan is rejected or breaks a rule.
``run_headless`` is the ``--port 0`` loop: ``run_async`` with a bounded wait, Python signal
handlers (Holoscan installs none), and ``request_stop`` once ``until`` (a
``DrainedCondition``) holds.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from ..ingest.affinity import CpuSets, keep_off_cores, parse_cpu_list
from ..settings import DAQIRI_INGEST_MODES
from .plan import Rule, check_plan, plan_summary

if TYPE_CHECKING:
    import argparse

    import holoscan as hs

    from .context import RunContext
    from .placement import Lanes

logger = logging.getLogger(__name__)


class ExitCode(enum.IntEnum):
    OK = 0
    RUN_FAILED = 1  # FAILED, timed out, or an error diagnostic
    USAGE = 2  # argparse's own status for bad flags
    PLAN_REJECTED = 3
    PLAN_RULES = 4
    EXPECTATION = 5  # ran cleanly, but an --expect-* check failed


def usage_error(message: str) -> NoReturn:
    """Reject a flag as argparse does: ``<prog>: error: <message>`` and exit ``USAGE``."""
    print(f"{Path(sys.argv[0]).name}: error: {message}", file=sys.stderr)
    raise SystemExit(ExitCode.USAGE)


def cpu_sets(args: argparse.Namespace) -> CpuSets:
    """The CPU sets the flags ask for, without changing the process: in DAQIRI modes the NIC
    lcores are reserved. An ``--ingest-cpus`` outside the CPUs left is a usage error."""
    in_daqiri_mode = getattr(args, "ingest", None) in DAQIRI_INGEST_MODES
    reserved = (args.nic_master_core, args.nic_rx_core) if in_daqiri_mode else ()
    drain_core = getattr(args, "nic_drain_core", -1)
    if getattr(args, "ingest", None) == "gpudirect" and drain_core >= 0:
        reserved += (drain_core,)  # the pinned native drain's core
    try:
        return CpuSets.compute(
            reserved=reserved,
            ingest=parse_cpu_list(args.ingest_cpus) if args.ingest_cpus else None,
            busy_poll=args.ingest_busy_poll,
        )
    except ValueError as exc:
        usage_error(str(exc))


def prepare_process(args: argparse.Namespace) -> CpuSets:
    """Interpreter, logging and CPU sets for this process; call it before any thread starts."""
    sys.setswitchinterval(0.001)  # Python threads hand the GIL over every 1 ms
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(threadName)s %(name)s: %(message)s"
    )
    cpus = cpu_sets(args)
    if cpus.helper != cpus.all:  # DAQIRI lcores reserved and/or a busy-polled ingest carve-out
        os.sched_setaffinity(0, cpus.helper)
        # sweep earlier threads off the lcores and the busy-polled CPUs
        keep_off_cores(cpus.all - cpus.helper)
    return cpus


def compile_plan(
    g: hs.Graph,
    lanes: Lanes,
    *,
    rules: Iterable[Rule] = (),
    ctx: RunContext | None = None,
    plan_json: str | None = None,
    topics_catalog: str | None = None,
) -> hs.Plan:
    """Compile ``g`` with the lanes' placement; exit with ``PLAN_REJECTED`` or ``PLAN_RULES``
    instead of starting a plan that cannot keep the composition's promises."""
    import holoscan as hs

    plan = hs.compile(g, lanes.compile_options())
    if not plan.ok:
        print(plan.summary(), file=sys.stderr)
        raise SystemExit(ExitCode.PLAN_REJECTED)
    if violations := check_plan(plan, rules):
        for violation in violations:
            print(f"plan rule violated: {violation}", file=sys.stderr)
        raise SystemExit(ExitCode.PLAN_RULES)
    if plan_json:
        _write(plan_json, plan.json())
    if topics_catalog:
        Path(topics_catalog).parent.mkdir(parents=True, exist_ok=True)  # save_json makes none
        plan.endpoint_catalog.save_json(topics_catalog)
    if ctx is not None:
        ctx.plan_summary = plan_summary(plan)
    for diagnostic in plan.warnings:
        logger.warning("plan: %s", _describe(diagnostic))
    return plan


def print_plan(plan: hs.Plan) -> int:
    """``--plan-only``: print the lanes, edges and clocks, start nothing."""
    print(json.dumps(plan_summary(plan), indent=1))
    return ExitCode.OK


def run_headless(
    plan: hs.Plan,
    ctx: RunContext,
    *,
    until: Callable[[], bool] | None = None,
    timeout_s: float = 0,
    report: str | None = None,
    expect: Callable[[], list[str]] | None = None,
) -> int:
    """Run ``plan`` until a signal, ``until()`` or ``timeout_s`` (0: none), on the main thread.

    Returns ``RUN_FAILED`` if the run failed, timed out or left an error diagnostic, else
    ``EXPECTATION`` if ``expect()`` reports anything, else ``OK``.
    """
    import holoscan as hs

    stop = threading.Event()
    previous = {
        sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    monitor = ctx.stall_monitor
    try:
        if monitor is not None:
            monitor.start()
        with hs.run_async(plan) as session:
            ctx.status.graph_status = "running"
            ctx.stop_reason = _wait_for_stop(session, stop, until, timeout_s)
            if ctx.stop_reason == "timeout":
                logger.error("not drained after %s s: %s", timeout_s, ctx.drained.why())
            session.request_stop()
            session.wait(10)
    finally:
        if monitor is not None:
            monitor.stop()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    ctx.status.from_session(session)
    if report:
        write_report(report, ctx, session)
    errors = report_diagnostics(session)
    for error in errors:
        logger.error("run: %s", error)
    if session.termination == hs.RunTermination.FAILED or ctx.stop_reason == "timeout" or errors:
        return ExitCode.RUN_FAILED
    if expect is not None and (unmet := expect()):
        for problem in unmet:
            logger.error("expectation not met: %s", problem)
        return ExitCode.EXPECTATION
    return ExitCode.OK


def _wait_for_stop(
    session: hs.RunSession,
    stop: threading.Event,
    until: Callable[[], bool] | None,
    timeout_s: float,
) -> str:
    """Why the run should stop: "signal", "drained", "timeout", or "terminated" when the
    session ended by itself."""
    deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
    while not session.wait(0.05):
        if stop.is_set():
            return "signal"
        if until is not None and until():
            return "drained"
        if deadline is not None and time.monotonic() >= deadline:
            return "timeout"
    return "terminated"


def report_diagnostics(session: hs.RunSession) -> list[str]:
    """The run's error-severity diagnostics (``MAX_START_DELAY_MISS`` is only a warning)."""
    import holoscan as hs

    error = hs.diagnostics.DiagnosticSeverity.ERROR
    return [_describe(d) for d in session.diagnostics if d.severity == error]


def write_report(path: str | None, ctx: RunContext, session: hs.RunSession) -> dict:
    """``ctx.report_dict(session)``, also written as JSON to ``path`` when one is given."""
    report = ctx.report_dict(session)
    if path:
        _write(path, json.dumps(report, indent=2, default=_json_scalar) + "\n")
    return report


def _json_scalar(value: object) -> object:
    # NumPy scalars and arrays in the stats objects (np.int64 is not an int)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _describe(diagnostic: hs.diagnostics.Diagnostic) -> str:
    """``TOKEN [operator]: message``; a failed callback's message is its traceback."""
    detail = diagnostic.detail
    error = detail.error if detail is not None else None
    message = error.message.strip() if error is not None and error.message else ""
    where = f" [{detail.operator_path}]" if detail is not None and detail.operator_path else ""
    return f"{diagnostic.token}{where}: {message}" if message else f"{diagnostic.token}{where}"


def _write(path: str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
