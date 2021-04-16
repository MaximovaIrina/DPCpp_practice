#include "blur.h"
#include <CL/sycl.hpp>
#include "DiviceSelector.h"

using namespace cl::sycl;

constexpr access::mode dp_read = access::mode::read;
constexpr access::mode dp_write = access::mode::write;

#define GPU "UHD Graphics"
#define CPU "Core"

void blurParallel(const std::vector<float>& src, std::vector<float>& dst,
  const int h, const int w, const int n, const float cpu_split, const int fs, double& time) {
  try {
    auto queue_property = property_list{ property::queue::enable_profiling() };
    DeviceSelector gpu_selector(GPU);
    DeviceSelector cpu_selector(CPU);
    queue q_gpu(gpu_selector, async_handler{}, queue_property);
    queue q_cpu(cpu_selector, async_handler{}, queue_property);

    std::cout << "GPU device: " << q_gpu.get_device().get_info<info::device::name>() << '\n';
    std::cout << "CPU device: " << q_cpu.get_device().get_info<info::device::name>() << '\n';

    buffer<float, 1> src_buf(src.data(), src.size());
    buffer<float, 1> dst_buf(dst.data(), dst.size());

    int split = h * cpu_split + fs;
    split -= split % fs;

    buffer<int, 1> fs_buf(&fs, 1);
    buffer<int, 1> w_buf(&w, 1);
    buffer<int, 1> h_buf(&h, 1);

    sycl::event e_cpu = q_cpu.submit([&](handler& cgh) {
      auto src_ = src_buf.get_access<dp_read>(cgh);
      auto w_ = w_buf.get_access<dp_read>(cgh);
      auto h_ = h_buf.get_access<dp_read>(cgh);
      auto fs_ = fs_buf.get_access<dp_read>(cgh);
      auto dst_ = dst_buf.get_access<dp_write>(cgh);

      cgh.parallel_for(n * w * (split + fs / 2), [=](nd_item<1> item) {
        int ofs = fs_[0] / 2;
        int h = h_[0];
        int w = w_[0];
        int h_cpu = item.get_global_range(0) / (3 * w);

        int id = item.get_global_id(0);
        int chn = id / (h_cpu * w);
        int pos = id % (h_cpu * w);
        int x = pos / w;
        int y = pos % w;

        float dst_px = 0;
        float divisor = ((x + ofs) - sycl::max(0, x - ofs)) *
          (sycl::min(w - 1, y + ofs) - sycl::max(0, y - ofs));

        for (int ii = sycl::max(0, x - ofs); ii <= x + ofs; ++ii)
          for (int jj = sycl::max(0, y - ofs); jj <= sycl::min(w - 1, y + ofs); ++jj)
            dst_px += src_[chn * h * w + ii * w + jj];

        dst_px /= divisor;
        if (x < h_cpu - fs / 2) dst_[chn * h * w + x * w + y] = dst_px;
        });
      });

    sycl::event e_gpu = q_gpu.submit([&](handler& cgh) {
      auto src_ = src_buf.get_access<dp_read>(cgh);
      auto fs_ = fs_buf.get_access<dp_read>(cgh);
      auto w_ = w_buf.get_access<dp_read>(cgh);
      auto h_ = h_buf.get_access<dp_read>(cgh);
      auto dst_ = dst_buf.get_access<dp_write>(cgh);

      cgh.parallel_for(n * w * (h - split + fs / 2), [=](nd_item<1> item) {
        int ofs = fs_[0] / 2;
        int h = h_[0];
        int w = w_[0];
        int h_gpu = item.get_global_range(0) / (3 * w);
        int h_cpu = h - h_gpu;

        int id = item.get_global_id(0);
        int chn = id / (h_gpu * w);
        int pos = id % (h_gpu * w);
        int x = pos / w + h_cpu;
        int y = pos % w;

        float divisor = (sycl::min(h - 1, x + ofs) - (x - ofs)) *
          (sycl::min(w - 1, y + ofs) - sycl::max(0, y - ofs));

        float dst_px = 0;
        for (int ii = x - ofs; ii <= sycl::min(h, x + ofs); ++ii)
          for (int jj = sycl::max(0, y - ofs); jj <= sycl::min(w - 1, y + ofs); ++jj)
            dst_px += src_[chn * h * w + ii * w + jj];

        dst_px /= divisor;
        dst_[chn * h * w + x * w + y] = dst_px;
        });
      });

    e_gpu.wait_and_throw();
    e_cpu.wait_and_throw();

    q_gpu.wait_and_throw();
    q_cpu.wait_and_throw();


    double time_gpu = 1e-9 * (e_gpu.get_profiling_info<sycl::info::event_profiling::command_end>() -
      e_gpu.get_profiling_info<sycl::info::event_profiling::command_start>());
    double time_cpu = 1e-9 * (e_cpu.get_profiling_info<sycl::info::event_profiling::command_end>() -
      e_cpu.get_profiling_info<sycl::info::event_profiling::command_start>());

    std::cout << "\n";
    std::cout << "Time GPU: " << time_gpu << std::endl;
    std::cout << "Time CPU: " << time_cpu << std::endl;


    time = std::max(time_gpu, time_cpu);
  }
  catch (invalid_parameter_error& E) {
    std::cout << E.what() << std::endl;
    std::cout << "With OpenCL error code : " << E.get_cl_code() << std::endl;
  }
}


void blurScalar(const std::vector<float>& src, std::vector<float>& dst,
  const int h, const int w, const int n, const int fs, double& time) {
  const int ofs = fs / 2;
  float pxl = 0.f;
  float divisor = 0.f;

  auto start = std::chrono::steady_clock::now();
  for (int c = 0; c < n; ++c)
    for (int i = 0; i < h; ++i)
      for (int j = 0; j < w; ++j) {
        pxl = 0.f;
        for (int ii = std::max(0, i - ofs); ii < std::min(h, i + ofs); ++ii)
          for (int jj = std::max(0, j - ofs); jj < std::min(w, j + ofs); ++jj)
            pxl += src[c * h * w + ii * w + jj];

        divisor = (std::min(h - 1, i + ofs) - std::max(0, i - ofs)) *
          (std::min(w - 1, j + ofs) - std::max(0, j - ofs));

        dst[c * h * w + i * w + j] = pxl / divisor;
      }
  auto end = std::chrono::steady_clock::now();

  time = std::chrono::duration<double>(end - start).count();
}


bool blurCheck(const std::vector<float>& a, const std::vector<float>& b) {
  for (size_t i = 0; i < 2160 * 3840; ++i)
    if (a[i] - b[i] >= 1e-3)
      return false;
  return true;
}
