#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fetch the FIX SBE reference tool (sbe-all, the real-logic/aeron SBE compiler) into the
# gitignored .cache/sbe, pinned by version and SHA-256, for the optional schema check
# (tests/python/library/test_sbe_tool.py; see docs/VALIDATION.md). Needs curl; the check
# itself needs java (>= 17) and g++. Nothing in the repository depends on it at run time.
set -euo pipefail
VERSION="${SBE_TOOL_VERSION:-1.41.0}"
SHA256="${SBE_TOOL_SHA256:-729b4271beb577f380380a1793e60f156ff75859b08ea603c4e2ae6b5e9a04ca}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$REPO/.cache/sbe"
JAR="$DIR/sbe-all-$VERSION.jar"
mkdir -p "$DIR"
if [ ! -f "$JAR" ]; then
    curl -fsSL --retry 3 -o "$JAR.part" \
        "https://repo1.maven.org/maven2/uk/co/real-logic/sbe-all/$VERSION/sbe-all-$VERSION.jar"
    mv "$JAR.part" "$JAR"
fi
echo "$SHA256  $JAR" | sha256sum -c - >/dev/null || { echo "fetch_sbe_tool: checksum mismatch for $JAR" >&2; rm -f "$JAR"; exit 1; }
echo "$JAR"
