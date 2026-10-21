# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print the risk summaries ``equity_analytics --topics`` exports over holomq, one per line.

    python -m holoscan_finance.tools.topic_tail --catalog PATH [--seconds N] [--timeout-s 5]

An external subscriber of the ``finance.risk`` topic (``RiskSummaryOp``), found through the
endpoint catalog the app saves at ``--topics-catalog``. It must share the engine's IPC
namespace: ``docker exec`` into the engine container, or a container started with
``--ipc=container:<engine>``; never ``--ipc=host``. A paused or slow tail costs the graph
nothing: the export is latest-only, so its summaries are dropped there. ``--timeout-s`` is the
attach bound and may not exceed the export's (``edges.TOPIC_MAX_WAIT_MS``): beyond about 5 s
without an explicit bound, attaching fails with ``ENDPOINT_ATTACH_BOUND_INVALID``.
"""

from __future__ import annotations

import argparse
import time

import holoscan as hs
from holoscan import channel

from ..operators.edges import TOPIC_MAX_WAIT_MS

TOPIC = "finance.risk"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m holoscan_finance.tools.topic_tail", description=__doc__.splitlines()[0]
    )
    p.add_argument("--catalog", required=True, help="endpoint catalog (--topics-catalog)")
    p.add_argument("--seconds", type=float, default=0, help="stop after this long (0: at EOS)")
    p.add_argument(
        "--timeout-s",
        type=float,
        default=5,
        help=f"attach bound, at most {TOPIC_MAX_WAIT_MS / 1000:g} s",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    a = p.parse_args(argv)
    if not 0 < a.timeout_s <= TOPIC_MAX_WAIT_MS / 1000:
        p.error(f"--timeout-s must be within (0, {TOPIC_MAX_WAIT_MS / 1000:g}]")
    catalog = hs.topics.EndpointCatalog.load(a.catalog)
    deadline = time.monotonic() + a.seconds if a.seconds > 0 else None
    with channel.Runtime() as runtime:
        sub = runtime.subscriber(catalog, TOPIC, timeout=hs.time.ms(round(a.timeout_s * 1000)))
        while deadline is None or time.monotonic() < deadline:
            try:
                sample = sub.try_receive()
            except hs.errors.ClosedError:  # the engine stopped: end of stream
                break
            if sample is None:
                time.sleep(0.01)
                continue
            print(sample.data.data, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
