# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Market-data protocol and compute library; no Holoscan, CuPy only inside functions.

The FIX SBE codec and schema (``sbe_codec``), observation records (``records``), Ethernet/UDP framing
and pcap files (``framing``), CPU and GPU quote bars (``bars``, ``gpu_bars``), the host
sequence rules (``sequence``), the bars processor (``bar_builder``), the GPU SBE kernels
and decoder (``gpu_sbe``, ``gpu_decoder``), the covariance engine (``covariance``) and
kernel warm-up (``warmup``).
"""
