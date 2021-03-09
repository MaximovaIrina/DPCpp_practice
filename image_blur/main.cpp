#include <iostream>
#include "image.h"
#include "blur.h"


#define GPU "UHD Graphics"
#define CPU "Core"
#define NUM_CH 3;

constexpr size_t filter_size = 15;


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
  if (!img::Load("forest.jpg", src_img))
    std::cout << "Image load error" << std::endl;
  int h = src_img.h;
  int w = src_img.w;
  int n = NUM_CH;

  std::vector<float> src = imgToVec(src_img);

  std::vector<float> dst_scalar(src.size(), 0);
  std::vector<float> dst_CPU(src.size(), 0);
  std::vector<float> dst_GPU(src.size(), 0);
  
  double time_scalar = 0.;
  double time_CPU = 0.;
  double time_GPU = 0.;

  blurScalar(src, dst_scalar, h, w, n, filter_size, time_scalar);
  blurParallel(CPU, src, dst_CPU, h, w, n, filter_size, time_CPU);
  blurParallel(GPU, src, dst_GPU, h, w, n, filter_size, time_GPU);

  std::cout << "\nSEQ time:\t" << time_scalar << "s\n";
  std::cout << "CPU time:\t" << time_CPU << "s (boost: " << time_scalar / time_CPU << ")\n";
  std::cout << "GPU time:\t" << time_GPU << "s (boost: " << time_scalar / time_GPU << ")\n";

  std::string statusAdd_CPU = blurCheck(dst_scalar, dst_CPU) ? "success" : "fail";
  std::string statusAdd_GPU = blurCheck(dst_scalar, dst_GPU) ? "success" : "fail";
  std::cout << "\nBlur status: CPU - " << statusAdd_CPU << ", GPU - " << statusAdd_GPU << std::endl;

  img::Image img_scalar = vecToImg(dst_scalar, h, w);
  img::Image img_CPU = vecToImg(dst_CPU, h, w);
  img::Image img_GPU = vecToImg(dst_GPU, h, w);

  if (!img::SavePng("Scalar.jpg", img_scalar))
    std::cout << "Image save error" << std::endl;
  if (!img::SavePng("CPU.jpg", img_CPU))
    std::cout << "Image save error" << std::endl;
  if (!img::SavePng("GPU.jpg", img_GPU))
    std::cout << "Image save error" << std::endl;
  
  return 0;
}