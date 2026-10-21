# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``CallbackSinkOp``: hand every item to a callable, in order, on the producer's lane.

Ports: ``items`` (in, any pyobject). Edges: ``lossless(8)`` from risk, ``lossless(2)`` from
the cross-check, ``lossless(64)`` from bars when every window matters (bench, tests). Lane:
the producer's; fed only from another lane it would be that lane's second entry operator.
Results publish as soon as risk emits them. An exception from the callback fails the run;
callbacks that must not stop the graph (``ResultsStore.on_risk``, ``on_comparison``) catch
internally.
"""

from __future__ import annotations

from collections.abc import Callable

import holoscan as hs

from ..ref import Ref


class CallbackSinkOp(hs.Operator):
    """``callback(item)`` for every item received."""

    items = hs.Input("items", hs.pyobject)

    callback = hs.Param(Ref(), description="Ref(callable(item)); required")

    def __init__(self) -> None:
        self._callback: Callable[[object], object] | None = None

    def setup(self, spec: hs.OperatorSpec) -> None:
        if not callable(self.callback.obj):
            raise ValueError(f"callback must be Ref(callable), got {self.callback!r}")
        spec.input(self.items)

    def contract(self) -> hs.Contract:
        return hs.Contract().trigger(hs.OnEach(self.items))

    def start(self) -> None:
        self._callback = self.callback.obj

    def compute(self, ctx: hs.ExecutionContext) -> None:
        self._callback(self.items.receive_data())
