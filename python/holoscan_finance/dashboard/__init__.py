# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The web dashboard: the HTTP/JSON contract over the control and telemetry objects.

``model.DashboardModel`` builds ``/snapshot`` from host objects only (no CUDA, no data-plane
lock); ``server.make_app`` maps the endpoints onto it and ``server.serve`` runs the
graph under uvicorn (FastAPI lifespan owns the session). Nothing is imported here, so the
model stays importable without FastAPI.
"""
