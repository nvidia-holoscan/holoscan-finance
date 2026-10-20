// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Batch frame-address fetch for DAQIRI bursts. DAQIRI's Python binding exposes
// get_packet_ptr/get_packet_length one packet at a time (~1 us of Python each);
// this writes every packet's address and length for a burst in one call, with
// the GIL released. It accepts the BurstParams objects returned by the daqiri
// module, so it must be built with the same pybind11 internals ABI.
#include <cstdint>
#include <stdexcept>

#include "daqiri/common.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

static int64_t burst_frames(daqiri::BurstParams* burst,
                            py::array_t<uint64_t, py::array::c_style> out) {
  const int64_t n = daqiri::get_num_packets(burst);
  if (out.ndim() != 1 || out.shape(0) < 2 * n) {
    throw std::invalid_argument("out must hold 2 * num_packets uint64 entries");
  }
  uint64_t* o = out.mutable_data();
  {
    py::gil_scoped_release release;
    for (int64_t i = 0; i < n; ++i) {
      o[i] = reinterpret_cast<uintptr_t>(daqiri::get_packet_ptr(burst, static_cast<int>(i)));
      o[n + i] = daqiri::get_packet_length(burst, static_cast<int>(i));
    }
  }
  return n;
}

PYBIND11_MODULE(_daqiri_frames, m) {
  py::module_::import("daqiri");  // BurstParams must already be registered
  m.def("burst_frames", &burst_frames, py::arg("burst"), py::arg("out"),
        "Write packet addresses, then lengths, of a DAQIRI burst into out[0:2n]; return n.");
}
