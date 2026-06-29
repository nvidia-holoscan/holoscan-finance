# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file is entered via add_holohub_package(holoscan-finance ...) in
# pkg/CMakeLists.txt, which only fires when PKG_holoscan_finance is ON.
include(holohub_configure_deb)

install(
    FILES
        "${PROJECT_SOURCE_DIR}/metadata.json"
        "${PROJECT_SOURCE_DIR}/README.md"
        "${PROJECT_SOURCE_DIR}/LICENSE"
    DESTINATION "share/holoscan-finance")

holohub_configure_deb(
    NAME        "holoscan-finance"
    DESCRIPTION "${PROJECT_DESCRIPTION}"
    VERSION     "${PROJECT_VERSION}"
    VENDOR      "NVIDIA"
    CONTACT     "holohub@nvidia.com"
    DEPENDS     "holoscan (>= 4.3.0)"
)
