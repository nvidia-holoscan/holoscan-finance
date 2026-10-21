# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pure-Python Holoscan module. CMake is only glue for holoscan-cli: `holoscan build|run|test`
# always configure and build this project, and `holoscan run` puts <build>/python/lib on
# PYTHONPATH. Nothing is compiled; the native DAQIRI drain is built at run time, and only in
# DAQIRI images, because it must match the installed DAQIRI and pybind11 ABI.
cmake_minimum_required(VERSION 3.24)

project(holoscan_finance
    VERSION 0.2.0
    DESCRIPTION "Streaming NYQ equity analytics on Holoscan SDK 5"
    LANGUAGES NONE)

if(NOT DEFINED FINANCE_BUILD_TESTING)
    set(FINANCE_BUILD_TESTING ${PROJECT_IS_TOP_LEVEL}
        CACHE BOOL "Register the module's pytest suite with CTest")
endif()

# --- Stage the canonical package for build-tree imports -----------------------
# <build>/python/lib/holoscan_finance -> ../../../../python/holoscan_finance (relative
# symlink, valid on the host and inside the dev container), plus a minimal dist-info whose
# entry point makes Holoscan Core publish holoscan.finance. The distribution name matches
# pyproject.toml, so an installed wheel and the build tree never register the alias twice
# (importlib.metadata de-duplicates by name).
set(FINANCE_PYTHON_ROOT "${CMAKE_BINARY_DIR}/python/lib")
file(MAKE_DIRECTORY "${FINANCE_PYTHON_ROOT}")
file(RELATIVE_PATH _pkg_rel "${FINANCE_PYTHON_ROOT}" "${PROJECT_SOURCE_DIR}/python/holoscan_finance")
file(REMOVE "${FINANCE_PYTHON_ROOT}/holoscan_finance")
file(CREATE_LINK "${_pkg_rel}" "${FINANCE_PYTHON_ROOT}/holoscan_finance" SYMBOLIC)
set(_dist "${FINANCE_PYTHON_ROOT}/holoscan_finance-${PROJECT_VERSION}.dist-info")
file(GLOB _stale LIST_DIRECTORIES true "${FINANCE_PYTHON_ROOT}/holoscan_finance-*.dist-info")
if(_stale)
    file(REMOVE_RECURSE ${_stale})
endif()
file(WRITE "${_dist}/METADATA"
    "Metadata-Version: 2.1\nName: holoscan-finance\nVersion: ${PROJECT_VERSION}\n")
file(WRITE "${_dist}/entry_points.txt" "[holoscan.modules]\nfinance = holoscan_finance\n")

if(FINANCE_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
