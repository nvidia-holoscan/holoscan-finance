# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rolling-risk analytics: the processors of ``RollingRiskOp`` and ``RiskCrossCheckOp``.

``rolling.RollingRisk`` analyzes the newest bar window at a bounded rate, inline or in a
spawned worker; ``crosscheck.CpuCrossCheck`` recomputes validated GPU windows on the CPU;
``workers`` holds the spawn-worker functions (4.3 ``cpu_window``/``gpu_window``) and
``agreement`` the CPU/GPU comparison rules. Spawn-safe: a worker that imports this package
pulls in NumPy only (CuPy, SciPy and threadpoolctl load inside functions).
"""
