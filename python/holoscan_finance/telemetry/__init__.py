# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Telemetry the operators write and the dashboard, reports and finite runs read.

``stats`` (the ingest, bars and risk stats objects), ``capture`` (exact latency captures)
and ``stalls`` (the heartbeat and GC stall monitor). Nothing is imported here: the modules
belong to different work packages and must import independently.
"""
