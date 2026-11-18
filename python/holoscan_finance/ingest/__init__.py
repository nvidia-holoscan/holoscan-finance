# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packet receive library behind the four source operators; no Holoscan.

``affinity`` (CPU sets and thread masks), ``daqiri`` (binding checks and the RX config),
``accounting`` (receive accounting, one implementation for host and GPU paths), ``drains``
(the device ring and the native-protocol drains), ``gpu_ingest`` (the processor of both GPU
sources), ``host_receivers`` (the socket and DAQIRI host receivers) and ``native`` (the
GIL-free C++ drain). Nothing is imported here, so ``affinity`` stays stdlib-only to import.
"""
