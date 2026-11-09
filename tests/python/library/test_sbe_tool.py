# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The schema and the codec's reading of it, checked against the FIX SBE reference tool.

``sbe-tool`` (uk.co.real-logic:sbe-all) validates ``sbe_schema.xml`` against the SBE 1.0
schema rules with warnings fatal and generates a C++ codec; a probe compiled against that
codec prints the layout it encodes (identifiers, block lengths, every field's offset, length,
id and null value, the enum values, the price exponent), which must equal what
``sbe_codec`` derived from the XML on its own. Runs when ``java``, ``g++`` and the jar are
present (``scripts/fetch_sbe_tool.sh``, or ``SBE_TOOL_JAR``); skipped otherwise.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from holoscan_finance.marketdata import sbe_codec as sbe

REPO = Path(__file__).resolve().parents[3]
PROBE = r"""
#include <cstdio>
#include "holoscan_finance_sbe/MarketDataIncrementalRefresh.h"
using namespace holoscan_finance::sbe;
typedef MarketDataIncrementalRefresh M;
typedef M::NoMDEntries E;
int main() {
    std::printf("{\"schemaId\": %u, \"version\": %u, \"templateId\": %u, \"rootBlockLength\": %u,\n",
                (unsigned)M::sbeSchemaId(), (unsigned)M::sbeSchemaVersion(),
                (unsigned)M::sbeTemplateId(), (unsigned)M::sbeBlockLength());
    std::printf("\"headerLength\": %u, \"groupSizeLength\": %u, \"entryBlockLength\": %u, \"groupId\": %u,\n",
                (unsigned)MessageHeader::encodedLength(), (unsigned)GroupSizeEncoding::encodedLength(),
                (unsigned)E::sbeBlockLength(), (unsigned)M::NoMDEntriesId());
    std::printf("\"seqNum\": [%u, %u, %u], \"priceExponent\": %d, \"priceNull\": %lld, \"priceLength\": %u,\n",
                (unsigned)M::seqNumId(), (unsigned)M::seqNumEncodingOffset(), (unsigned)M::seqNumEncodingLength(),
                (int)PriceNULL4::exponent(), (long long)PriceNULL4::mantissaNullValue(),
                (unsigned)PriceNULL4::encodedLength());
    std::printf("\"entries\": {");
    std::printf("\"TransactTime\": [%u, %u, %u], ", (unsigned)E::transactTimeId(), (unsigned)E::transactTimeEncodingOffset(), (unsigned)E::transactTimeEncodingLength());
    std::printf("\"BidPx\": [%u, %u, %u], ", (unsigned)E::bidPxId(), (unsigned)E::bidPxEncodingOffset(), (unsigned)PriceNULL4::encodedLength());
    std::printf("\"OfferPx\": [%u, %u, %u], ", (unsigned)E::offerPxId(), (unsigned)E::offerPxEncodingOffset(), (unsigned)PriceNULL4::encodedLength());
    std::printf("\"LastPx\": [%u, %u, %u], ", (unsigned)E::lastPxId(), (unsigned)E::lastPxEncodingOffset(), (unsigned)PriceNULL4::encodedLength());
    std::printf("\"BidSize\": [%u, %u, %u, %u], ", (unsigned)E::bidSizeId(), (unsigned)E::bidSizeEncodingOffset(), (unsigned)E::bidSizeEncodingLength(), (unsigned)E::bidSizeNullValue());
    std::printf("\"OfferSize\": [%u, %u, %u, %u], ", (unsigned)E::offerSizeId(), (unsigned)E::offerSizeEncodingOffset(), (unsigned)E::offerSizeEncodingLength(), (unsigned)E::offerSizeNullValue());
    std::printf("\"LastQty\": [%u, %u, %u, %u], ", (unsigned)E::lastQtyId(), (unsigned)E::lastQtyEncodingOffset(), (unsigned)E::lastQtyEncodingLength(), (unsigned)E::lastQtyNullValue());
    std::printf("\"SecurityID\": [%u, %u, %u], ", (unsigned)E::securityIDId(), (unsigned)E::securityIDEncodingOffset(), (unsigned)E::securityIDEncodingLength());
    std::printf("\"EntryType\": [%u, %u, %u], ", (unsigned)E::entryTypeId(), (unsigned)E::entryTypeEncodingOffset(), (unsigned)E::entryTypeEncodingLength());
    std::printf("\"SecurityTradingStatus\": [%u, %u, %u]}, ", (unsigned)E::securityTradingStatusId(), (unsigned)E::securityTradingStatusEncodingOffset(), (unsigned)E::securityTradingStatusEncodingLength());
    std::printf("\"EntryType\": {\"Quote\": %u, \"Trade\": %u, \"TradingStatus\": %u}, ",
                (unsigned)EntryType::Quote, (unsigned)EntryType::Trade, (unsigned)EntryType::TradingStatus);
    std::printf("\"SecurityTradingStatus\": {\"TradingHalt\": %u, \"ReadyToTrade\": %u, \"NULL_VALUE\": %u}}\n",
                (unsigned)SecurityTradingStatus::TradingHalt, (unsigned)SecurityTradingStatus::ReadyToTrade,
                (unsigned)SecurityTradingStatus::NULL_VALUE);
    return 0;
}
"""


def _jar() -> str | None:
    env = os.environ.get("SBE_TOOL_JAR")
    if env:
        return env
    found = sorted(glob.glob(str(REPO / ".cache" / "sbe" / "sbe-all-*.jar")))
    return found[-1] if found else None


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=300, check=False, **kw)


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> Path:
    jar = _jar()
    if jar is None or not Path(jar).is_file():
        pytest.skip("sbe-tool jar not found (scripts/fetch_sbe_tool.sh or SBE_TOOL_JAR)")
    if shutil.which("java") is None or shutil.which("g++") is None:
        pytest.skip("java and g++ are needed for the sbe-tool check")
    out = tmp_path_factory.mktemp("sbe") / "cpp"
    run = _run(
        [
            "java",
            "-Dsbe.target.language=cpp",
            f"-Dsbe.output.dir={out}",
            "-Dsbe.validation.stop.on.error=true",
            "-Dsbe.validation.warnings.fatal=true",
            "-jar",
            jar,
            str(sbe.SCHEMA_PATH),
        ]
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert (out / "holoscan_finance_sbe" / "MarketDataIncrementalRefresh.h").is_file()
    return out


def test_sbe_tool_validates_the_schema_and_agrees_on_the_layout(generated, tmp_path):
    probe = tmp_path / "probe.cpp"
    probe.write_text(PROBE)
    build = _run(
        ["g++", "-std=c++17", "-O0", f"-I{generated}", str(probe), "-o", str(tmp_path / "probe")]
    )
    assert build.returncode == 0, build.stderr
    tool = json.loads(_run([str(tmp_path / "probe")]).stdout)
    s = sbe.SCHEMA
    assert (tool["schemaId"], tool["version"], tool["templateId"]) == (
        s.schema_id,
        s.version,
        s.template_id,
    )
    assert tool["rootBlockLength"] == s.root.size == sbe.ROOT_BLOCK_LENGTH
    assert tool["headerLength"] == s.header.size == sbe.HEADER.size
    assert tool["groupSizeLength"] == s.group_size.size == sbe.GROUP_SIZE.size
    assert tool["entryBlockLength"] == s.entries.size == sbe.ENTRY_BLOCK_LENGTH
    assert tool["groupId"] == s.group_id
    seq = s.root["SeqNum"]
    assert tool["seqNum"] == [seq.id, seq.offset, seq.size]
    assert tool["priceExponent"] == s.price_exponent and tool["priceNull"] == sbe.PRICE_NULL
    assert tool["priceLength"] == 8  # the constant exponent takes no space
    for field in s.entries.fields:
        got = tool["entries"][field.name]
        assert got[:3] == [field.id, field.offset, field.size], field.name
        if len(got) == 4:
            assert got[3] == field.null, field.name
    assert tool["EntryType"] == {
        k.split(".")[1]: v for k, v in s.constants.items() if k.startswith("EntryType.")
    }
    assert tool["SecurityTradingStatus"] == {
        "TradingHalt": sbe.TRADING_HALT,
        "ReadyToTrade": sbe.READY_TO_TRADE,
        "NULL_VALUE": sbe.STATUS_NULL,
    }


def test_generated_codec_decodes_the_golden_bytes(generated, tmp_path):
    """The reference codec reads the hand-transcribed golden packet as the codec does."""
    from library.test_sbe_codec import GOLDEN

    src = tmp_path / "decode.cpp"
    program = r"""
#include <cstdio>
#include <cstring>
#include <cstdint>
#include "holoscan_finance_sbe/MarketDataIncrementalRefresh.h"
using namespace holoscan_finance::sbe;
int main() {
    static unsigned char buf[] = {GOLDEN_BYTES};
    MessageHeader hdr(reinterpret_cast<char*>(buf), 0, sizeof buf, 0);
    MarketDataIncrementalRefresh m(reinterpret_cast<char*>(buf), hdr.encodedLength(), sizeof buf,
                                   hdr.blockLength(), hdr.version());
    if (hdr.templateId() != MarketDataIncrementalRefresh::sbeTemplateId() ||
        hdr.schemaId() != MarketDataIncrementalRefresh::sbeSchemaId()) return 2;
    std::printf("%llu", (unsigned long long)m.seqNum());
    MarketDataIncrementalRefresh::NoMDEntries &e = m.noMDEntries();
    while (e.hasNext()) {
        e.next();
        std::printf(" %llu %lld %lld %lld %u %u %u %u %u %u", (unsigned long long)e.transactTime(),
                    (long long)e.bidPx().mantissa(), (long long)e.offerPx().mantissa(),
                    (long long)e.lastPx().mantissa(), e.bidSize(), e.offerSize(), e.lastQty(),
                    (unsigned)e.securityID(), (unsigned)e.entryType(), (unsigned)e.securityTradingStatus());
    }
    std::printf(" %zu\n", (size_t)m.encodedLength());
    return 0;
}
"""
    src.write_text(program.replace("GOLDEN_BYTES", ",".join(str(b) for b in GOLDEN)))
    build = _run(["g++", "-std=c++17", f"-I{generated}", str(src), "-o", str(tmp_path / "decode")])
    assert build.returncode == 0, build.stderr
    run = _run([str(tmp_path / "decode")])
    assert run.returncode == 0
    fields = run.stdout.split()
    seq, entries = sbe.parse_packet(GOLDEN)
    e = entries[0]
    assert fields == [
        str(seq),
        str(int(e["TransactTime"])),
        str(int(e["BidPx"])),
        str(int(e["OfferPx"])),
        str(int(e["LastPx"])),
        str(int(e["BidSize"])),
        str(int(e["OfferSize"])),
        str(int(e["LastQty"])),
        str(int(e["SecurityID"])),
        str(int(e["EntryType"])),
        str(int(e["SecurityTradingStatus"])),
        str(len(GOLDEN) - sbe.HEADER.size),
    ]
    assert sys.byteorder == "little"  # the probe prints the buffer's native reading
