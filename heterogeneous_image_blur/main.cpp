#include <iostream>
#include "image.h"
#include "blur.h"

#define NUM_CH 3;

constexpr size_t filter_size = 15;
constexpr float cpu_boost = 4.5f;
constexpr float gpu_boost = 10.3f;

std::vector<float> imgToVec(const img::Image& img) {
  std::vector<float> v = img.r;
  v.insert(v.end(), img.g.begin(), img.g.end());
  v.insert(v.end(), img.b.begin(), img.b.end());
  return v;
}


img::Image vecToImg(const std::vector<float>& v, int h, int w) {
  img::Image img;
  img.r = std::vector<float>(v.begin(), v.begin() + h * w);
  img.g = std::vector<float>(v.begin() + h * w, v.begin() + 2 * h * w);
  img.b = std::vector<float>(v.begin() + 2 * h * w, v.end());
  img.h = h;
  img.w = w;
  return img;
}


int main() {
  img::Image src_img;
  if (!img::Load("forest.jpg", src_img)) std::cout << "Image load error" << std::endl;
  int h = src_img.h;
  int w = src_img.w;
  int n = NUM_CH;

  std::vector<float> src = imgToVec(src_img);

  std::vector<float> dst_scalar(src.size(), 0);
  std::vector<float> dst_parallel(src.size(), 0);
  
  double time_scalar{};
  double time_paralell{};

  float cpu_split = cpu_boost / (cpu_boost + gpu_boost);

  blurScalar(src, dst_scalar, h, w, n, filter_size, time_scalar);
  blurParallel(src, dst_parallel, h, w, n, cpu_split, filter_size, time_paralell);

  std::cout << "\nSEQ time:\t" << time_scalar << "s\n";
  std::cout << "Parallel time:\t" << time_paralell << "s (boost: " << time_scalar / time_paralell << ")\n";

  std::string status = blurCheck(dst_scalar, dst_parallel) ? "success" : "fail";
  std::cout << "\nBlur status: " << status << std::endl;

  img::Image img_scalar = vecToImg(dst_scalar, h, w);
  img::Image img_parallel = vecToImg(dst_parallel, h, w);

  if (!img::SavePng("Scalar.jpg", img_scalar)) std::cout << "Image save error" << std::endl;
  if (!img::SavePng("Parallel.jpg", img_parallel)) std::cout << "Image save error" << std::endl;
  
  return 0;
}