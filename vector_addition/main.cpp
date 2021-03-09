#include <iostream>
#include <vector>
#include "VectorsAddition.h"

#define CPU "Core"
#define GPU "UHD Graphics"

int main() {
  constexpr size_t array_size = static_cast<int>(1e6);
  std::cout << "Array size: " << array_size << '\n';
  
  std::vector<int> a(array_size, 0);
  std::vector<int> b(array_size, 0);
  std::vector<int> c_scalar(array_size, 0);
  std::vector<int> c_CPU(array_size, 0);
  std::vector<int> c_GPU(array_size, 0);

  vectorInit(a);
  vectorInit(b);

  double time_scalar = 0.;
  double time_CPU = 0.;
  double time_GPU = 0.;
  vectorAddScalar(a, b, c_scalar, time_scalar);
  vectorAddParallel(CPU, a, b, c_CPU, time_CPU);
  vectorAddParallel(GPU, a, b, c_GPU, time_GPU);

  std::cout << "\nSEQ time:\t" << time_scalar << "\n";
  std::cout << "CPU time:\t" << time_CPU << " (boost: " << time_scalar / time_CPU << ")\n";
  std::cout << "GPU time:\t" << time_GPU << " (boost: " << time_scalar / time_GPU << ")\n";

  std::string statusAdd_CPU = checkAdd(c_scalar, c_CPU) ? "success" : "fail";
  std::string statusAdd_GPU = checkAdd(c_scalar, c_GPU) ? "success" : "fail";
  std::cout << "\nAddition: CPU - " << statusAdd_CPU << ", GPU - " << statusAdd_GPU << std::endl;
  return 0;
}