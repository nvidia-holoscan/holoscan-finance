# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control plane: the objects HTTP and the replay loop use to steer a running graph.

``gate`` (the CPU cross-check gate), ``directory`` (feed manifests and generations),
``feed`` (sender pacing), ``fullday`` (full-day bursts), ``results`` (published analytics)
and ``status`` (the run state). They reach operators as ``Ref`` Params, never through
ports. Nothing is imported here: the modules belong to different work packages.
"""
