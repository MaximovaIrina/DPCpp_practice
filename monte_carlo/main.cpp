#include <iostream>
#include "monteCarlo.h"

#define GPU "UHD Graphics"
#define CPU "Core"

constexpr double res = 0.3868223;

int main() {
  bool repeat = true;
  while (repeat) {
    std::cout << "-------------------------------------------------------------------\n";
    int N, num_points_by_thread;
    std::cout << "N points: ";
    std::cin >> N;
    std::cout << "N thread points: ";
    std::cin >> num_points_by_thread;
    std::cout << "\n";

    float res_SEQ_std{};
    float res_SEQ_mkl{};
    float res_CPU{};
    float res_CPU_u{};

    std::pair<float, float> time_SEQ_std;
    std::pair<float, float> time_SEQ_mkl;
    std::pair<float, float> time_CPU;
    std::pair<float, float> time_CPU_u;

    monteCarlo_SEQ_std_rnd(N, res_SEQ_std, time_SEQ_std);
    monteCarlo_SEQ_mkl_rnd(N, res_SEQ_mkl, time_SEQ_mkl);
    monteCarlo_PLL(CPU, N, num_points_by_thread, res_CPU, time_CPU);

    std::cout << "\nTIME:\n";
    std::cout << "  SEQ_std\t" << time_SEQ_std.first + time_SEQ_std.second << "\t(gen: " << time_SEQ_std.first << ", alg: " << time_SEQ_std.second << ")\n";
    std::cout << "  SEQ_sobol\t" << time_SEQ_mkl.first + time_SEQ_mkl.second << "\t(gen: " << time_SEQ_mkl.first << ", alg: " << time_SEQ_mkl.second << ")\n";
    std::cout << "  CPU_sobol\t" << time_CPU.first + time_CPU.second << "\t(gen: " << time_CPU.first << ", alg: " << time_CPU.second << ", boost: " <<
      (time_SEQ_mkl.first + time_SEQ_mkl.second) / (time_CPU.first + time_CPU.second) << ")\n";


    std::cout << "\nACCURANCY:\n";
    std::cout << "  SEQ_std\t" << abs(res - res_SEQ_std) << "\n";
    std::cout << "  SEQ_sobol\t" << abs(res - res_SEQ_mkl) << "\n";
    std::cout << "  CPU_sobol\t" << abs(res - res_CPU) << "\n";

    std::cout << "\n\nRepeat?";
    std::cin >> repeat; 
  }
 
  return 0;
}