#include <CL/sycl.hpp>
#include "blur.h"
#include "DiviceSelector.h"

using namespace cl::sycl;

constexpr access::mode dp_read = access::mode::read;
constexpr access::mode dp_write = access::mode::write;


void blurParallel(const char* device, const std::vector<float>& src, std::vector<float>& dst,
                  const int h, const int w, const int n, const int fs, double& time) {  
  DeviceSelector selector(device);
  try {
    auto queue_property = property_list{ property::queue::enable_profiling() };
    queue q(selector, async_handler{}, queue_property);
    
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << '\n';

    range<1> img_shape{ n * h * w };
    buffer<float, 1> src_buf(src.data(), img_shape);
    buffer<float, 1> dst_buf(dst.data(), img_shape);
    buffer<int, 1> fs_buf(&fs, range<1> {1});
    buffer<int, 1> h_buf(&h, range<1> {1});
    buffer<int, 1> w_buf(&w, range<1> {1});

    sycl::event e = q.submit([&](handler& cgh) {
      auto src_ = src_buf.get_access<dp_read>(cgh);
      auto fs_ = fs_buf.get_access<dp_read>(cgh);
      auto h_ = h_buf.get_access<dp_read>(cgh);
      auto w_ = w_buf.get_access<dp_read>(cgh);
      auto dst_ = dst_buf.get_access<dp_write>(cgh);

      cgh.parallel_for(img_shape, [=](nd_item<1> item) {
        int ofs = fs_[0] / 2;
        int h = h_[0];
        int w = w_[0];

        int c = item.get_global_id(0) / (h * w);
        int i = item.get_global_id(0) % (h * w);
        int x = i / w;
        int y = i % w;

        float dst_px = 0;
        float divisor = (sycl::min(h - 1, x + ofs) - sycl::max(0, x - ofs)) *
                      (sycl::min(w - 1, y + ofs) - sycl::max(0, y - ofs));

        for (int ii = sycl::max(0, x - ofs); ii <= sycl::min(h - 1, x + ofs); ++ii)
          for (int jj = sycl::max(0, y - ofs); jj <= sycl::min(w - 1, y + ofs); ++jj)
            dst_px += src_[c * h * w + ii * w + jj];

        dst_px /= divisor;
        dst_[item.get_global_id(0)] = dst_px;
        });
      });

    e.wait_and_throw();
    q.wait_and_throw();

    double start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    double end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    time = 1e-9 * (end - start);
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
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] - b[i] >= 1) return false;
  return true;
}