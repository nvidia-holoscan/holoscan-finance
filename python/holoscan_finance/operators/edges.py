# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge presets that applications pass to ``add_flow``, so ``plan.json`` shows the intent.

``LATEST`` is the latest-only mailbox (``DROP_OLDEST``, depth 1): bars never block on
analytics, and a slow consumer skips windows. ``lossless(depth)`` is a ``BLOCKING`` edge of
explicit depth (the default depth 1 fails a run with ``BackpressuredError`` under load).
``topic_export_kwargs`` bounds a holomq export so a stalled external reader can never fail
the run; the edge into an export is always ``LATEST`` (a ``BLOCKING`` one ended a run with
``HOLOMQ_BACKPRESSURED`` after 17 emits once a reader stopped).
"""

from __future__ import annotations

import holoscan as hs

LATEST = {"queue_policy": hs.QueuePolicy.DROP_OLDEST, "queue_depth": 1}

# Subscribers must use a timeout <= this: one of 5.001 s fails without an attach bound.
TOPIC_MAX_WAIT_MS = 10_000


def lossless(depth: int) -> dict:
    """A ``BLOCKING`` edge of explicit depth."""
    return {"queue_policy": hs.QueuePolicy.BLOCKING, "queue_depth": int(depth)}


def topic_export_kwargs(max_wait_ms: int = TOPIC_MAX_WAIT_MS) -> dict:
    """Optional peer, an explicit attach bound, and bounded subscribers:
    ``export_topic(..., **topic_export_kwargs())``."""
    return {
        "attach": hs.topics.TopicAttachPolicy(
            peer=hs.topics.PeerRequirement.OPTIONAL, max_wait=hs.time.ms(max_wait_ms)
        ),
        "bounds": hs.topics.EndpointBounds(
            subscriber=hs.topics.SubscriberEndpointBounds(queue_depth=8, max_borrowed_samples=1)
        ),
    }
