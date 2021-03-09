#include <CL/sycl.hpp>

#include "VectorsAddition.h"
#include "DiviceSelector.h"

using namespace cl::sycl;

constexpr access::mode dp_read = access::mode::read;
constexpr access::mode dp_write = access::mode::write;


void vectorAddParallel(const char* device, const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c, double& time) {
  try {
    DeviceSelector selector(device);
    
    std::cout << "\nList of all available devices: \n";
    auto queue_property = property_list{ property::queue::enable_profiling() };
    queue q(selector, async_handler{}, queue_property);
    
    std::cout << "\nSelected device: " << q.get_device().get_info<info::device::name>() << '\n';

    range<1> size{ a.size() };
    buffer<int, 1> a_buf(a.data(), size);
    buffer<int, 1> b_buf(b.data(), size);
    buffer<int, 1> c_buf(c.data(), size);

    sycl::event e = q.submit([&](handler& cgh) {
      auto a_ = a_buf.get_access<dp_read>(cgh);
      auto b_ = b_buf.get_access<dp_read>(cgh);
      auto c_ = c_buf.get_access<dp_write>(cgh);
      sycl::stream out(128, 128, cgh);

      cgh.parallel_for(range<1>(a.size()), [=](nd_item<1> item) {
        size_t i = item.get_global_id(0);
        c_[i] = a_[i] + b_[i];
        if (i == 0) {
          out << "Number of work groups discovered: " << item.get_group_range(0) << sycl::endl;
          out << "Size of groups: " << item.get_local_range(0) << sycl::endl;
        }
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


void vectorAddScalar(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c, double& time) {
  auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < c.size(); i++)
    c[i] = a[i] + b[i];
  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration<double>(end - start).count();
}


void vectorInit(std::vector<int>& a) {
  for (size_t i = 0; i < a.size(); i++) a[i] = i;
}


bool checkAdd(std::vector<int>& sum_scalar, std::vector<int>& sum_parallel) {
  for (size_t i = 0; i < sum_parallel.size(); i++)
    if (sum_parallel[i] != sum_scalar[i]) return false;
  return true;
}