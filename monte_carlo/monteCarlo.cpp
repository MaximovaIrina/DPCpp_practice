#include <CL/sycl.hpp>
#include "oneapi/mkl/rng.hpp" 
#include "monteCarlo.h"
#include "DiviceSelector.h"
#include <iostream>
#include <ctime>
#include <random> 

using namespace cl::sycl;


void monteCarlo_PLL(const char* device, int N, int num_points_by_thread, float& res, std::pair<float, float>& time) {
  DeviceSelector selector(device);
  try {
    auto queue_property = property_list{ property::queue::enable_profiling() };
    queue q(selector, async_handler{}, queue_property);
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << '\n';

    int n = 0;
    size_t num_thread = ceil(N / num_points_by_thread);

    buffer<int, 1> buf_n(&n, 1);
    buffer<int, 1> buf_N(&N, 1);
    buffer<int, 1> buf_num_p_by_th(&num_points_by_thread, 1);

    std::vector<float> vec_points(3 * N);
    buffer<float, 1> buf_points(vec_points.data(), vec_points.size());
    
    oneapi::mkl::rng::sobol engine(q, 3);
    oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard> distribution(0.0f, 1.0f);

    try {
      auto start = std::chrono::steady_clock::now();
      oneapi::mkl::rng::generate(distribution, engine, vec_points.size(), buf_points);
      auto end = std::chrono::steady_clock::now();
      time.first = std::chrono::duration<double>(end - start).count();
    } catch (std::exception e) {
      std::cout << e.what() << std::endl;
    }

    sycl::event e = q.submit([&](handler& cgh) {
      auto N_points = buf_N.get_access<access::mode::read>(cgh);
      auto count = buf_num_p_by_th.get_access<access::mode::read>(cgh);
      auto points = buf_points.get_access<access::mode::read>(cgh);

      auto n_accessor = buf_n.get_access<access::mode::read_write>(cgh);
      auto n_sum_reduction = ONEAPI::reduction(n_accessor, 0, ONEAPI::plus<>());

      cgh.parallel_for(nd_range<1>{num_thread, 1}, n_sum_reduction, [=](nd_item<1> it, auto& n_sum) {
          int n_internal = 0;
          int start = it.get_global_linear_id() * count[0] * 3;
          int ind = start;
          float x = 0;
          float y = 0;
          float z = 0;

          while (ind < cl::sycl::min(start + count[0] * 3, N_points[0] * 3)) {
            x = points[ind];
            y = points[ind + 1];
            z = points[ind + 2];
            n_internal += (z < sin(x) * cos(y)) ? 1 : 0;
            ind += 3;
          }
          n_sum += n_internal;
        });
      });

    e.wait_and_throw();
    q.wait_and_throw();

    double start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    double end = e.get_profiling_info<sycl::info::event_profiling::command_end>();

    time.second = 1e-9 * (end - start);
    res = float(n) / float(N);
  }
  catch (invalid_parameter_error& E) {
    std::cout << E.what() << std::endl;
    std::cout << "With OpenCL error code : " << E.get_cl_code() << std::endl;
  }
}



void monteCarlo_SEQ_mkl_rnd(const int N, float& res, std::pair<float, float>& time) {
  std::vector<float> vec_points(3 * N);
  buffer<float, 1> buf_points(vec_points.data(), vec_points.size());

  queue q(cpu_selector{});
  oneapi::mkl::rng::sobol engine(q, 3);
  oneapi::mkl::rng::uniform<float> distribution(0.0f, 1.0f);

  int n = 0;
  float x = 0;
  float y = 0;
  float z = 0;

  auto start = std::chrono::steady_clock::now();
  oneapi::mkl::rng::generate(distribution, engine, vec_points.size(), buf_points);
  auto end = std::chrono::steady_clock::now();
  time.first = std::chrono::duration<double>(end - start).count();


  start = std::chrono::steady_clock::now();
  {
    for (int i = 0; i < vec_points.size(); i += 3) {
      x = vec_points[i];
      y = vec_points[i + 1];
      z = vec_points[i + 2];
      n += (z < sin(x)* cos(y)) ? 1 : 0;
    }
    res = float(n) / float(N);
  }
  end = std::chrono::steady_clock::now();
  time.second = std::chrono::duration<double>(end - start).count();
  return;
}


void monteCarlo_SEQ_std_rnd(const int N, float& res, std::pair<float, float>& time) {
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(std::time(0)));
  std::vector<float> vec_points(3 * N);
  int n = 0;
  float x = 0;
  float y = 0;
  float z = 0;

  auto start = std::chrono::steady_clock::now();
  for (auto& point : vec_points) point = static_cast<double>(gen()) / static_cast<double>(gen.max());
  auto end = std::chrono::steady_clock::now();
  time.first = std::chrono::duration<double>(end - start).count();

  start = std::chrono::steady_clock::now();
  {
    for (int i = 0; i < N; i += 3) {
      x = vec_points[i];
      y = vec_points[i+1];
      z = vec_points[i+2];
      n += (z < sin(x)* cos(y)) ? 1 : 0;
    }
    res = float(n) / float(N);
  }
  end = std::chrono::steady_clock::now();
  time.second = std::chrono::duration<double>(end - start).count();
  return;
}