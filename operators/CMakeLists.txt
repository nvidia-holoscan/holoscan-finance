# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared finance contracts and CSV helpers are staged before operator modules.
configure_file(
    ${PROJECT_SOURCE_DIR}/python/holoscan/finance/models.py
    ${FINANCE_PYTHON_PKG_DIR}/models.py COPYONLY)
configure_file(
    ${PROJECT_SOURCE_DIR}/python/holoscan/finance/io.py
    ${FINANCE_PYTHON_PKG_DIR}/io.py COPYONLY)
configure_file(
    ${PROJECT_SOURCE_DIR}/python/holoscan/finance/py.typed
    ${FINANCE_PYTHON_PKG_DIR}/py.typed COPYONLY)

add_holohub_operator(market_data_replay_op)
add_holohub_operator(rolling_statistics_op)
add_holohub_operator(portfolio_risk_op)

configure_file(
    ${PROJECT_SOURCE_DIR}/python/holoscan/finance/__init__.py
    ${FINANCE_PYTHON_PKG_DIR}/__init__.py COPYONLY)

# scikit-build-core installs directly into a wheel's site-packages root. A
# regular CMake/CPack install uses the Holoscan SDK prefix and therefore needs
# the conventional python/lib subdirectory beneath that prefix.
if(DEFINED SKBUILD)
    set(_finance_python_install_dest ".")
else()
    set(_finance_python_install_dest "python/lib")
endif()
install(
    DIRECTORY "${FINANCE_PYTHON_ROOT}/holoscan"
    DESTINATION "${_finance_python_install_dest}"
    PATTERN "__pycache__" EXCLUDE)
