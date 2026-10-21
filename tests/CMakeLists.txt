# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Two CTest entries over one pytest tree, split by tier (markers in pyproject.toml). SDK tests
# skip without `import holoscan` and GPU tests without a CUDA device; `slow` and `timing`
# tests run only through pytest -m, never from CTest.
find_package(Python3 REQUIRED COMPONENTS Interpreter)

function(finance_pytest tier markers)
    add_test(NAME python.${tier}
        COMMAND "${Python3_EXECUTABLE}" -m pytest -q -m "${markers}" tests/python
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    # Prepend the staged build tree (it carries the holoscan.finance alias) and keep the
    # image's PYTHONPATH, where the SDK lives.
    set_tests_properties(python.${tier} PROPERTIES
        ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:${FINANCE_PYTHON_ROOT}"
        LABELS "python;${tier}")
endfunction()

finance_pytest(unit "not gpu and not slow")
finance_pytest(gpu "gpu and not slow")
