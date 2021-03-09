#pragma once

void vectorInit(std::vector<int>& a);

void vectorAddParallel(const char* device, const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c, double& time);

void vectorAddScalar(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c, double& time);

bool checkAdd(std::vector<int>& sum_scalar, std::vector<int>& sum_parallel);

