# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authored lanes, their thread placement and the compile options, in one place.

The lane rule (J1 plus convexity), for whoever edits a composition: each lane is weakly
connected and convex (no path leaves it and comes back), and has exactly one entry
operator, either clock-triggered or fed only from other lanes; every other member has an
input edge from inside the lane, and may have more from other lanes. Two entries are
``PARTITION_COVER_INVALID``; a non-convex layout fails with ``PARTITION_ANCHOR_NON_CONVEX``
once anything is partitioned automatically (``graphs/test_lane_rule.py``). Every operator
goes into a lane, so nothing is partitioned automatically. ``ingest`` runs on
``CpuSets.ingest`` (busy-polled only with ``--ingest-busy-poll``), every other lane on
``CpuSets.helper``, so no lane runs on a DAQIRI lcore (4.3 ``_isolate_cores`` and
``MultiThreadScheduler`` workers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..ingest.affinity import CpuSets

if TYPE_CHECKING:
    import holoscan as hs

INGEST = "ingest"


class Lanes:
    """Authored partitions, their ``ThreadPlacement`` and the ``CompileOptions``."""

    def __init__(self, cpus: CpuSets, *, ingest_busy_poll: bool = False) -> None:
        self.cpus = cpus
        self.ingest_busy_poll = ingest_busy_poll
        self._lanes: dict[str, list[hs.operators.OperatorHandle]] = {}
        self._applied = False

    def add(self, name: str, *ops: hs.operators.OperatorHandle) -> Lanes:
        """Put ``ops`` in lane ``name``; a repeated name extends the lane."""
        if self._applied:
            raise RuntimeError(f"lane {name!r}: the partitions were already applied")
        self._lanes.setdefault(name, []).extend(ops)
        return self

    def apply(self, g: hs.Graph) -> None:
        """Author every lane on ``g``, one ``g.partition(name)`` each (a second one raises)."""
        if self._applied:
            raise RuntimeError("the lanes were already applied")
        for name, ops in self._lanes.items():
            g.partition(name).add(*ops)
        self._applied = True

    def compile_options(self) -> hs.CompileOptions:
        """``ingest`` on ``cpus.ingest``, every other lane on ``cpus.helper``; split
        serialization domains are intended (each lane is its own runtime thread)."""
        import holoscan as hs

        ex = hs.execution
        deployment = ex.DeploymentBinding()
        for name in self._lanes:
            if name == INGEST:
                wake = (
                    ex.WakeMode.BUSY_POLL if self.ingest_busy_poll else ex.WakeMode.SLEEPING_WAITSET
                )
                placement = ex.ThreadPlacement(
                    affinity=ex.CPUAffinity(sorted(self.cpus.ingest)), wake_mode=wake
                )
            else:
                placement = ex.ThreadPlacement(affinity=ex.CPUAffinity(sorted(self.cpus.helper)))
            deployment.bind_thread_placement(name, placement)
        return hs.CompileOptions(
            deployment=deployment,
            serialization_domain_splits=ex.SerializationDomainSplitPolicy.ALLOW,
        )

    @property
    def names(self) -> list[str]:
        return list(self._lanes)

    def members(self, name: str) -> list[str]:
        """Operator names of lane ``name``, in the order they were added."""
        return [op.name for op in self._lanes[name]]
