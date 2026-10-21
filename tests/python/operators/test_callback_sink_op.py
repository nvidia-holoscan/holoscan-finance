# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``CallbackSinkOp``: every item reaches the callable in order; a missing or non-callable
callback is an authoring error; a callback exception fails the run with its traceback."""

from __future__ import annotations

import holoscan as hs
import pytest
from fakes.stores import ResultsStoreFake
from holoscan.testing import OperatorTestHarness

from holoscan_finance.operators import CallbackSinkOp
from holoscan_finance.ref import Ref

pytestmark = pytest.mark.sdk


def test_items_reach_the_callback_in_order():
    store = ResultsStoreFake()
    with OperatorTestHarness(CallbackSinkOp, params={"callback": Ref(store.on_risk)}) as h:
        h.push_many("items", list(range(200)))
        assert h.run().ok
    assert store.risk == list(range(200))


@pytest.mark.parametrize("callback", [Ref(), Ref(42)])
def test_a_missing_or_non_callable_callback_is_rejected_at_authoring(callback):
    g = hs.Graph("sink", default_clock=hs.RealtimeClock)
    with pytest.raises(hs.errors.HoloscanError, match="callback must be Ref"):
        g.op(CallbackSinkOp, "publish", callback=callback)


def test_a_callback_exception_fails_the_run():
    def boom(item: object) -> None:
        raise RuntimeError(f"analytics consumer broke on {item}")

    with OperatorTestHarness(CallbackSinkOp, params={"callback": Ref(boom)}) as h:
        h.push("items", 7)
        result = h.run(check=False)
    assert result.termination is hs.RunTermination.FAILED
    (failure,) = [d for d in result.diagnostics if d.token == "RUNTIME_OPERATOR_CALLBACK_FAILED"]
    assert "analytics consumer broke on 7" in failure.detail.error.message
