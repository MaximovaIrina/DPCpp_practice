#pragma once

void monteCarlo_PLL(const char* device, int N, int num_points_by_thread, float& res, std::pair<float, float>& time);

void monteCarlo_SEQ_mkl_rnd(const int N, float& res, std::pair<float, float>& time);

void monteCarlo_SEQ_std_rnd(const int N, float& res, std::pair<float, float>& time);