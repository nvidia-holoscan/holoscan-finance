# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The equity replay feed: observations mapped to SBE entries, its manifest, tapes and test data.

``mapping`` (the replay protocol and the Python reference decoder), ``manifest`` (the feed
directory a replay generation announces), ``host_decoder`` (the processor of
``SbeDecodeOp``), ``tape`` (prepared observation tapes) and ``synthetic`` (deterministic
tapes and captures for tests and NIC-free validation). No Holoscan, no CuPy.
"""
