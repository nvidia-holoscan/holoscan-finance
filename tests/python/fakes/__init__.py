# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protocol fakes shared by the tests: DAQIRI, the native drain and the results store.

Each module belongs to the work package whose interface it stands in for (``fake_daqiri``
and ``fake_drain``: ingest; ``stores``: analytics). Tests import them as ``fakes.<module>``
because ``tests/python`` is on the path; none of them is a pytest fixture.
"""
