#pragma once
#include <vector>

void blurParallel(const std::vector<float>& src, std::vector<float>& dst,
  const int h, const int w, const int n, const float cpu_split, const int fs, double& time);

void blurScalar(const std::vector<float>& src, std::vector<float>& dst,
  const int h, const int w, const int n, const int fs, double& time);

bool blurCheck(const std::vector<float>& a, const std::vector<float>& b);