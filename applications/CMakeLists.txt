# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Each application is gated on its own APP_<name> option, defaulting to ${BUILD_ALL}.
# Operator dependencies are force-enabled when the application is selected.
add_holohub_application(intraday_risk_pipeline
    DEPENDS OPERATORS
        market_data_replay_op
        rolling_statistics_op
        portfolio_risk_op
)
