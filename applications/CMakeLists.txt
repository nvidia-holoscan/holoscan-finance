# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Each application is gated on its own APP_<name> option, defaulting to ${BUILD_ALL}.
# Operator dependencies are force-enabled when the application is selected.
add_holohub_application(equity_analytics
    DEPENDS OPERATORS gpu_mitch_source_op gpu_quote_bars_op gpu_rolling_risk_op)
