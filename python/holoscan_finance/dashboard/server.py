# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP for the dashboard and the replay tools, and the uvicorn run loop.

``make_app`` keeps the 4.3 endpoints, bodies and 422 mapping (``ValueError``,
``TypeError``, ``KeyError``) and adds ``GET /plan`` and ``GET /healthz``. Handlers are
``async def`` (no thread-pool hop per request: measured sub-ms tails under load); only the
blocking full-day start and the GIL fault injection go to worker threads. ``serve`` is
pattern B: uvicorn owns the main thread and SIGINT, the FastAPI lifespan owns the
``RunSession`` (stopped in about 1 ms on shutdown, but waited for without a bound, so the
report and exit code always see a stopped graph). The port is bound before the graph
starts, and uvicorn's own exit statuses never leak. From 4.3 ``web_app`` and ``main``.
FastAPI is imported here so the pure-tier HTTP tests can use ``make_app``; Holoscan only
inside ``serve``.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import logging
import os
import signal
import socket
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .model import DashboardModel

logger = logging.getLogger(__name__)

INDEX_HTML = Path(__file__).parent / "webui" / "index.html"
# Shutdown warns if the graph has not stopped after this long, then keeps waiting.
STOP_WARNING_S = 10.0


def _mutate(fn: Callable[[Any], Any], payload: Any) -> Any:
    try:
        return fn(payload)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, detail=str(exc)) from exc


def make_app(model: DashboardModel, ctx: Any, *, lifespan: Any = None) -> FastAPI:
    """The dashboard's FastAPI app over ``model`` and the control objects of ``ctx``.
    ``lifespan`` (used by ``serve``) owns the graph session."""
    app = FastAPI(title="NYQ equity price, volatility and correlation", lifespan=lifespan)
    # Handlers carry no return annotation: one would make FastAPI validate every snapshot
    # against a response model instead of 4.3's plain JSON encoding.

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return INDEX_HTML.read_text(encoding="utf-8")

    @app.get("/snapshot")
    async def snapshot():
        return model.snapshot()

    @app.get("/plan")
    async def plan():
        return ctx.plan_summary

    @app.get("/healthz")
    async def healthz():
        return {"graph_status": ctx.status.graph_status, "termination": ctx.status.termination}

    @app.post("/manifest")
    async def manifest(payload: dict):
        return _mutate(ctx.directory.set_manifest, payload)

    @app.post("/capture")
    async def capture(payload: dict):
        return _mutate(model.start_capture, payload)

    @app.get("/capture")
    async def capture_status():
        return ctx.tap.status() or {}

    @app.post("/capture/finish")
    async def capture_finish():
        return _mutate(lambda _: model.finish_capture(), {})

    @app.post("/replay/fullday")
    async def fullday(payload: dict):
        def start(p: dict) -> dict:
            return ctx.fullday.start(int(p.get("pps", 200_000)))

        # Waits up to 60 s for the replay controller: never on the event loop.
        return await anyio.to_thread.run_sync(_mutate, start, payload)

    @app.post("/replay/resume")
    async def resume(payload: dict):
        """Back to the paced sender: unpause and restart it at the session open (new generation)."""
        return _mutate(lambda _: ctx.feed.resume(), payload)

    @app.get("/feedrate")
    async def feedrate():
        return ctx.feed.control()

    @app.post("/feedrate")
    async def control(payload: dict):
        return _mutate(ctx.feed.control, payload)

    @app.post("/feedstatus")
    async def status(payload: dict):
        return ctx.feed.sender_status(payload)

    if os.environ.get("HOLOSCAN_FINANCE_FAULTS") == "1":

        @app.post("/debug/hold_gil")
        async def hold_gil(payload: dict):
            """Fault injection: hold the interpreter lock for ``seconds``, as a Python thread
            blocked in the kernel does (the full-day stalls). ctypes.PyDLL keeps the GIL."""
            seconds = min(float(payload.get("seconds", 1.0)), 10.0)
            await anyio.to_thread.run_sync(ctypes.PyDLL(None).usleep, int(seconds * 1e6))
            return {"held_s": seconds}

    return app


@contextlib.contextmanager
def _signals_return_here() -> Iterator[None]:
    """uvicorn handles SIGINT/SIGTERM and, after its graceful shutdown, raises the signal
    again under the handler it found. With a no-op handler that ends ``server.run()``
    normally; SIGTERM's default action would kill the process before the report is written."""
    if threading.current_thread() is not threading.main_thread():
        yield  # uvicorn installs no handlers off the main thread either
        return
    previous = {sig: signal.signal(sig, lambda *_: None) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def serve(
    plan: Any,
    ctx: Any,
    *,
    host: str,
    port: int,
    exit_when_drained: bool = False,
    report: str | None = None,
    expect: Callable[[], list[str]] | None = None,
) -> int:
    """Run ``plan`` behind the dashboard until SIGINT/SIGTERM (or, with
    ``exit_when_drained``, until ``ctx.drained()``); returns an ``ExitCode``."""
    import holoscan as hs
    import uvicorn

    from ..app.runner import ExitCode, report_diagnostics, write_report

    sessions: list[Any] = []

    async def watch(session: Any) -> None:
        # Mirrors the session for /snapshot and /healthz; a finite run ends the server.
        try:
            while True:
                await asyncio.sleep(0.25)
                ctx.status.from_session(session)
                if not session.is_running:
                    # It ended by itself (failed, or completed). Without --exit-when-drained
                    # the dashboard stays up to show why, as 4.3's did.
                    ctx.stop_reason = ctx.stop_reason or "terminated"
                    if exit_when_drained:
                        server.should_exit = True
                    return
                if exit_when_drained and ctx.drained():
                    ctx.stop_reason = "drained"
                    session.request_stop()
                    server.should_exit = True
                    return
        except Exception:
            logger.exception("dashboard watcher failed; stopping")
            server.should_exit = True

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            # As in run_headless, leaving the block waits for the stop without a bound: slow
            # operator stop()s (pool shutdowns, DAQIRI) finish before the report is written.
            with hs.run_async(plan) as session:
                sessions.append(session)
                ctx.status.graph_status = "running"
                if ctx.stall_monitor is not None:
                    ctx.stall_monitor.start()
                watcher = asyncio.create_task(watch(session))
                try:
                    yield
                finally:
                    watcher.cancel()
                    ctx.stop_reason = ctx.stop_reason or "signal"
                    session.request_stop()
                    if not session.wait(STOP_WARNING_S):
                        logger.warning(
                            "graph still stopping after %g s; waiting for operator stop()",
                            STOP_WARNING_S,
                        )
        finally:
            if ctx.stall_monitor is not None:
                ctx.stall_monitor.stop()

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:  # before the lifespan starts the graph: a busy port must not start it
        sock = socket.create_server((host, port), family=family)
    except OSError as exc:
        logger.error("dashboard cannot listen on %s:%d: %s", host, port, exc)
        return ExitCode.RUN_FAILED
    app = make_app(DashboardModel(ctx), ctx, lifespan=lifespan)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    logger.info("dashboard on http://%s:%d", host, port)
    with sock, _signals_return_here():
        try:
            server.run(sockets=[sock])
        except SystemExit as exc:
            # uvicorn exits when the lifespan fails to start (run_async raised); its status
            # 3 would read as our PLAN_REJECTED.
            logger.error("dashboard failed to start (uvicorn exit status %s)", exc.code)
            return ExitCode.RUN_FAILED
    if not sessions:  # the lifespan never started the graph
        return ExitCode.RUN_FAILED
    session = sessions[0]
    ctx.status.from_session(session)
    if report:
        write_report(report, ctx, session)
    errors = report_diagnostics(session)
    for error in errors:
        logger.error("run: %s", error)
    if session.termination == hs.RunTermination.FAILED or errors:
        return ExitCode.RUN_FAILED
    if expect is not None and (unmet := expect()):
        for msg in unmet:
            logger.error("expectation not met: %s", msg)
        return ExitCode.EXPECTATION
    return ExitCode.OK
