# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# --- holoscan.marketdata: the modules the operators use; each operator copies itself in flat ---
foreach(_md_mod mitch.py gpu_mitch.py daqiri.py wire.py bars.py gpu_bars.py covariance.py)
    configure_file(
        ${PROJECT_SOURCE_DIR}/python/holoscan/marketdata/${_md_mod}
        ${MARKETDATA_PYTHON_PKG_DIR}/${_md_mod} COPYONLY)
endforeach()
file(COPY ${PROJECT_SOURCE_DIR}/python/holoscan/marketdata/native
     DESTINATION ${MARKETDATA_PYTHON_PKG_DIR}
     PATTERN "__pycache__" EXCLUDE)

add_holohub_operator(gpu_mitch_source_op)
add_holohub_operator(gpu_quote_bars_op)
add_holohub_operator(gpu_rolling_risk_op)

configure_file(
    ${PROJECT_SOURCE_DIR}/python/holoscan/marketdata/__init__.py
    ${MARKETDATA_PYTHON_PKG_DIR}/__init__.py COPYONLY)

# --- holoscan.equities: the NYQ application package (every module, so none is missed) ---
file(GLOB _eq_mods CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/python/holoscan/equities/*.py)
foreach(_eq_mod ${_eq_mods})
    get_filename_component(_eq_name ${_eq_mod} NAME)
    configure_file(${_eq_mod} ${EQUITIES_PYTHON_PKG_DIR}/${_eq_name} COPYONLY)
endforeach()
file(COPY ${PROJECT_SOURCE_DIR}/python/holoscan/equities/webui
     DESTINATION ${EQUITIES_PYTHON_PKG_DIR}
     PATTERN "__pycache__" EXCLUDE)

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
