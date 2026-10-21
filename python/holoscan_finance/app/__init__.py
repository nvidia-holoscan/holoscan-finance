# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plumbing shared by the applications, so an application file is only its composition.

``cli`` (flags and their checks), ``context`` (the control plane and telemetry objects of a
run), ``placement`` (authored lanes and their CPU placement), ``plan`` (rules checked on
``plan.json()`` before a run), ``runner`` (process set-up, compile, headless run, report)
and ``drained`` (when a finite run has nothing left in flight); ``bench`` is
``market_data_bench``'s. The modules import Holoscan only inside functions, and this
package imports none of them.
"""
