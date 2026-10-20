// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// GIL-free GPUDirect drain for GpuMitchSourceOp. A native thread takes DAQIRI bursts,
// launches the gather_frame kernel (compiled by CuPy, passed in as a CUfunction) into a
// device slot ring, copies the per-packet receive header to pinned host memory and
// returns the burst to the NIC. Python reads one record per burst and releases ring
// space after decoding. A Python thread that stalls while holding the interpreter lock
// (for example one blocked in the kernel page allocator) no longer stops the NIC
// buffers from being recycled; the ring absorbs the stall instead.
//
// Steady state performs no heap or CUDA allocation (a new CUDA allocation can wake a
// runtime-suspended Thor iGPU for 0.1-0.6 s) and waits on the stream by polling. When
// the ring is full the thread holds the burst until Python releases space, so overflow
// shows up as NIC drops (sequence gaps), never as silently discarded packets.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <pthread.h>
#include <time.h>

#include "daqiri/common.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace {

constexpr int kFields = 9;  // start, n, recv_ns, t0, t_frames, t_launch, t_sync, t_free, wait_ns
constexpr int kHeaderCols = 7;
constexpr std::array<int64_t, 3> kGapMs{10, 50, 100};

int64_t clock_ns(clockid_t id) {
  timespec ts;
  clock_gettime(id, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

std::string cu_error(const char* what, CUresult r) {
  const char* name = nullptr;
  cuGetErrorName(r, &name);
  return std::string(what) + " failed: " + (name ? name : std::to_string(static_cast<int>(r)));
}

class Drain {
 public:
  Drain(int port, int queue, uintptr_t gather_fn, int device, uintptr_t slots,
        uintptr_t header_dev, uintptr_t header_host, int64_t capacity, int stride,
        uintptr_t packed_dev, uintptr_t packed_host, int64_t max_burst, int min_payload,
        int64_t max_records)
      : port_(port), queue_(queue), fn_(reinterpret_cast<CUfunction>(gather_fn)), device_(device),
        slots_(slots), header_dev_(header_dev),
        header_host_(reinterpret_cast<int64_t*>(header_host)), capacity_(capacity),
        stride_(stride), packed_dev_(packed_dev),
        packed_host_(reinterpret_cast<uint64_t*>(packed_host)), max_burst_(max_burst),
        min_payload_(min_payload), records_(static_cast<size_t>(max_records) * kFields),
        max_records_(max_records) {
    if (capacity <= 0 || max_burst <= 0 || max_burst > capacity || max_records <= 0) {
      throw std::invalid_argument("bad ring sizes");
    }
  }

  ~Drain() { stop(); }

  void start() {
    if (thread_.joinable()) throw std::runtime_error("drain already started");
    stop_.store(false);
    running_.store(true);
    last_rx_ns_.store(clock_ns(CLOCK_MONOTONIC));
    thread_ = std::thread([this] { run(); });
  }

  void stop() {
    stop_.store(true);
    if (thread_.joinable()) {
      py::gil_scoped_release release;
      thread_.join();
    }
  }

  // All records published since the last call: int64 (k, kFields).
  py::array_t<int64_t> take() {
    const int64_t head = rec_head_.load(std::memory_order_acquire);
    const int64_t tail = rec_tail_.load(std::memory_order_relaxed);
    const int64_t k = head - tail;
    py::array_t<int64_t> out({k, static_cast<int64_t>(kFields)});
    int64_t* o = out.mutable_data();
    for (int64_t i = 0; i < k; ++i) {
      const size_t at = static_cast<size_t>((tail + i) % max_records_) * kFields;
      std::memcpy(o + i * kFields, &records_[at], kFields * sizeof(int64_t));
    }
    rec_tail_.store(head, std::memory_order_release);
    return out;
  }

  // Python has finished with every slot before ring position ``end``.
  void release(int64_t end) {
    int64_t cur = tail_.load(std::memory_order_relaxed);
    if (end > cur) tail_.store(end, std::memory_order_release);
  }

  void reset_peak() {
    peak_.store(0);
    poll_gap_max_ns_.store(0);
    for (auto& n : poll_gaps_) n.store(0);
  }

  py::dict stats() const {
    py::dict d;
    d["running"] = running_.load();
    d["bursts"] = bursts_.load();
    d["packets"] = packets_.load();
    d["ring_peak"] = peak_.load();
    d["fill"] = head_seen_.load() - tail_.load();
    d["poll_gap_max_ms"] = poll_gap_max_ns_.load() / 1e6;  // longest time between NIC polls
    d["poll_gap_at"] = poll_gap_at_ns_.load();
    for (size_t i = 0; i < kGapMs.size(); ++i) {  // poll gaps over 10, 50 and 100 ms
      d[("poll_gaps_" + std::to_string(kGapMs[i]) + "ms").c_str()] = poll_gaps_[i].load();
    }
    d["ring_wait_ms"] = ring_wait_ns_.load() / 1e6;
    d["ring_waits"] = ring_waits_.load();
    d["last_rx_ns"] = last_rx_ns_.load();
    d["host_ns"] = host_ns_.load();
    std::lock_guard<std::mutex> lock(error_mu_);
    d["error"] = error_;
    return d;
  }

 private:
  void fail(const std::string& message) {
    std::lock_guard<std::mutex> lock(error_mu_);
    if (error_.empty()) error_ = message;
  }

  bool check(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return true;
    fail(cu_error(what, r));
    return false;
  }

  void run() {
    pthread_setname_np(pthread_self(), "hf-drain");  // visible in /proc/<pid>/task/*/comm
    CUcontext ctx = nullptr;
    CUstream stream = nullptr;
    CUdevice dev;
    if (check(cuInit(0), "cuInit") && check(cuDeviceGet(&dev, device_), "cuDeviceGet") &&
        check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain") &&
        check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent") &&
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate")) {
      loop(stream);
    }
    if (stream) cuStreamDestroy(stream);
    if (ctx) cuDevicePrimaryCtxRelease(dev);
    running_.store(false);
  }

  void loop(CUstream stream) {
    using daqiri::Status;
    int64_t last_poll = clock_ns(CLOCK_MONOTONIC);
    while (!stop_.load(std::memory_order_relaxed)) {
      const int64_t poll = clock_ns(CLOCK_MONOTONIC);
      const int64_t gap = poll - last_poll;
      if (gap > poll_gap_max_ns_.load(std::memory_order_relaxed)) {
        poll_gap_max_ns_.store(gap, std::memory_order_relaxed);
        poll_gap_at_ns_.store(poll, std::memory_order_relaxed);
      }
      for (size_t i = 0; i < kGapMs.size(); ++i) {
        if (gap > kGapMs[i] * 1000000LL) poll_gaps_[i].fetch_add(1, std::memory_order_relaxed);
      }
      last_poll = poll;
      daqiri::BurstParams* burst = nullptr;
      const Status st = daqiri::get_rx_burst(&burst, port_, queue_);
      if (st != Status::SUCCESS || burst == nullptr) {
        if (st != Status::NULL_PTR && st != Status::NOT_READY && st != Status::SUCCESS) {
          fail("get_rx_burst returned status " + std::to_string(static_cast<int>(st)));
          return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        continue;
      }
      const int64_t t0 = clock_ns(CLOCK_MONOTONIC);
      const int64_t recv_ns = clock_ns(CLOCK_REALTIME);
      last_rx_ns_.store(t0, std::memory_order_relaxed);
      const int64_t n = daqiri::get_num_packets(burst);
      if (n <= 0) {
        daqiri::free_all_packets_and_burst_rx(burst);
        continue;
      }
      if (n > max_burst_) {
        daqiri::free_all_packets_and_burst_rx(burst);
        fail("burst of " + std::to_string(n) + " packets exceeds max_burst");
        return;
      }
      // A burst never wraps: skip to the ring start if it would.
      int64_t pos = head_;
      if (pos % capacity_ + n > capacity_) pos += capacity_ - pos % capacity_;
      // Ring or record queue full: keep the burst until Python releases space.
      int64_t wait_ns = 0;
      const int64_t w0 = clock_ns(CLOCK_MONOTONIC);
      bool waited = false;
      while (pos + n - tail_.load(std::memory_order_acquire) > capacity_ ||
             rec_head_.load(std::memory_order_relaxed) -
                     rec_tail_.load(std::memory_order_acquire) >= max_records_) {
        if (stop_.load(std::memory_order_relaxed)) {
          daqiri::free_all_packets_and_burst_rx(burst);
          return;
        }
        waited = true;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
      if (waited) {
        wait_ns = clock_ns(CLOCK_MONOTONIC) - w0;
        ring_wait_ns_.fetch_add(wait_ns);
        ring_waits_.fetch_add(1);
      }
      const int64_t slot = pos % capacity_;
      for (int64_t i = 0; i < n; ++i) {
        packed_host_[i] =
            reinterpret_cast<uintptr_t>(daqiri::get_packet_ptr(burst, static_cast<int>(i)));
        packed_host_[n + i] = daqiri::get_packet_length(burst, static_cast<int>(i));
      }
      const int64_t t_frames = clock_ns(CLOCK_MONOTONIC);
      CUdeviceptr packed = packed_dev_;
      int n32 = static_cast<int>(n), min_payload = min_payload_, stride = stride_;
      CUdeviceptr slots = slots_ + static_cast<CUdeviceptr>(slot) * stride_;
      CUdeviceptr header = header_dev_ + static_cast<CUdeviceptr>(slot) * kHeaderCols * 8;
      void* args[] = {&packed, &n32, &min_payload, &slots, &stride, &header};
      bool ok = check(cuMemcpyHtoDAsync(packed, packed_host_, 2 * n * 8, stream), "copy frames") &&
                check(cuLaunchKernel(fn_, n32, 1, 1, 128, 1, 1, 0, stream, args, nullptr),
                      "gather_frame launch") &&
                check(cuMemcpyDtoHAsync(header_host_ + slot * kHeaderCols, header,
                                        n * kHeaderCols * 8, stream),
                      "copy header");
      const int64_t t_launch = clock_ns(CLOCK_MONOTONIC);
      CUresult q = CUDA_ERROR_NOT_READY;
      while (ok && (q = cuStreamQuery(stream)) == CUDA_ERROR_NOT_READY) {
      }
      ok = ok && check(q, "gather_frame");
      const int64_t t_sync = clock_ns(CLOCK_MONOTONIC);
      daqiri::free_all_packets_and_burst_rx(burst);  // payloads copied: recycle NIC buffers
      const int64_t t_free = clock_ns(CLOCK_MONOTONIC);
      if (!ok) return;
      head_ = pos + n;
      head_seen_.store(head_, std::memory_order_relaxed);
      const int64_t fill = head_ - tail_.load(std::memory_order_relaxed);
      if (fill > peak_.load(std::memory_order_relaxed)) peak_.store(fill);
      const int64_t r = rec_head_.load(std::memory_order_relaxed);
      int64_t* rec = &records_[static_cast<size_t>(r % max_records_) * kFields];
      const int64_t fields[kFields] = {pos, n, recv_ns, t0, t_frames, t_launch, t_sync, t_free,
                                       wait_ns};
      std::memcpy(rec, fields, sizeof(fields));
      rec_head_.store(r + 1, std::memory_order_release);
      bursts_.fetch_add(1, std::memory_order_relaxed);
      packets_.fetch_add(n, std::memory_order_relaxed);
      host_ns_.fetch_add(t_free - t0 - wait_ns, std::memory_order_relaxed);
    }
  }

  const int port_, queue_;
  const CUfunction fn_;
  const int device_;
  const CUdeviceptr slots_, header_dev_;
  int64_t* const header_host_;
  const int64_t capacity_;
  const int stride_;
  const CUdeviceptr packed_dev_;
  uint64_t* const packed_host_;
  const int64_t max_burst_;
  const int min_payload_;
  std::vector<int64_t> records_;  // ring of max_records_ x kFields, allocated once
  const int64_t max_records_;

  std::thread thread_;
  std::atomic<bool> stop_{false}, running_{false};
  int64_t head_ = 0;  // next free ring position (drain thread only)
  std::atomic<int64_t> tail_{0};
  std::atomic<int64_t> rec_head_{0}, rec_tail_{0};
  std::atomic<int64_t> bursts_{0}, packets_{0}, peak_{0}, ring_wait_ns_{0}, ring_waits_{0};
  std::atomic<int64_t> last_rx_ns_{0}, host_ns_{0}, head_seen_{0};
  std::atomic<int64_t> poll_gap_max_ns_{0}, poll_gap_at_ns_{0};
  std::array<std::atomic<int64_t>, 3> poll_gaps_{};  // counts over each kGapMs threshold
  mutable std::mutex error_mu_;
  std::string error_;
};

}  // namespace

PYBIND11_MODULE(_daqiri_drain, m) {
  py::module_::import("daqiri");  // DAQIRI must be initialized by its own module
  py::class_<Drain>(m, "Drain")
      .def(py::init<int, int, uintptr_t, int, uintptr_t, uintptr_t, uintptr_t, int64_t, int,
                    uintptr_t, uintptr_t, int64_t, int, int64_t>(),
           py::arg("port"), py::arg("queue"), py::arg("gather_fn"), py::arg("device"),
           py::arg("slots"), py::arg("header_dev"), py::arg("header_host"), py::arg("capacity"),
           py::arg("stride"), py::arg("packed_dev"), py::arg("packed_host"),
           py::arg("max_burst"), py::arg("min_payload"), py::arg("max_records"))
      .def("start", &Drain::start)
      .def("stop", &Drain::stop)
      .def("take", &Drain::take)
      .def("release", &Drain::release, py::arg("end"))
      .def("reset_peak", &Drain::reset_peak)
      .def("stats", &Drain::stats);
  m.attr("FIELDS") = py::make_tuple("start", "n", "recv_ns", "t0", "t_frames", "t_launch",
                                    "t_sync", "t_free", "wait_ns");
}
