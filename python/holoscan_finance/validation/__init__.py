# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bit-identity oracles and run verdicts for the GPU chain.

``compare`` decodes captures on the host and compares decoders field by field; its
``HostReference`` replays a capture through the host bars to give every closed bar of a
run. ``checkers`` pairs the GPU and CPU windows of one graph, compares the final bar state
of both backends and applies the bench gates to a report. NumPy only at import time.
"""
