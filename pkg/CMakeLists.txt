# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# add_holohub_package() handles the full cascade for both standalone
# packaging (`./holohub package holoscan-finance --pkg-generator DEB`, which
# sets -DPKG_holoscan_finance=ON via the HoloHub CLI) and in-tree
# HoloHub-monorepo packaging once this module is fetched as an external
# dependency:
#
#   1. Defines option(PKG_holoscan_finance ... ${BUILD_ALL}) — ON
#      when BUILD_ALL=ON (default standalone) or when the CLI passes the
#      option explicitly.
#   2. add_subdirectory(holoscan-finance) — enters
#      pkg/holoscan-finance/ where holohub_configure_deb() emits
#      CPackConfig-holoscan-finance.cmake.
#   3. set("OP_<x>" ON CACHE BOOL ... FORCE) and the equivalent for
#      APPLICATIONS — cascades to the operator and application subprojects
#      so the .deb gets actual content. pkg/ is add_subdirectory()'d
#      *before* operators/ and applications/, so those see the forced ON
#      value when their own helpers run option().
add_holohub_package(holoscan-finance
                    OPERATORS
                        gpu_mitch_source_op
                        gpu_quote_bars_op
                        gpu_rolling_risk_op
                    APPLICATIONS equity_analytics)
