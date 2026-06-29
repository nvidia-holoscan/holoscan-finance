# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Stage both the Python entry point and its deterministic fixture. Keeping the
# data path relative to the staged script makes source-tree and build-tree runs
# behave identically.
add_custom_target(python_intraday_risk_pipeline ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/intraday_risk_pipeline.py"
        "${CMAKE_CURRENT_BINARY_DIR}/intraday_risk_pipeline.py"
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${CMAKE_CURRENT_BINARY_DIR}/../data"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/../data/synthetic_market_data.csv"
        "${CMAKE_CURRENT_BINARY_DIR}/../data/synthetic_market_data.csv"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/intraday_risk_pipeline.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/../data/synthetic_market_data.csv"
    BYPRODUCTS
        "${CMAKE_CURRENT_BINARY_DIR}/intraday_risk_pipeline.py"
        "${CMAKE_CURRENT_BINARY_DIR}/../data/synthetic_market_data.csv"
    COMMENT "Staging the intraday risk pipeline"
    VERBATIM
)

# Reference applications are source examples for wheels, but the Debian
# package is a complete module distribution under /opt/nvidia/holoscan.
if(NOT DEFINED SKBUILD)
    install(
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/intraday_risk_pipeline.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/metadata.json"
        DESTINATION "applications/intraday_risk_pipeline/python")
    install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/../data/synthetic_market_data.csv"
        DESTINATION "applications/intraday_risk_pipeline/data")
    install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/../README.md"
        DESTINATION "applications/intraday_risk_pipeline")
endif()
