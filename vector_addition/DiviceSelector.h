#pragma once
#include <CL/sycl.hpp>


class DeviceSelector : public cl::sycl::device_selector {
  std::string device;

public:
  explicit DeviceSelector(std::string _device) : device(_device) {}

  int operator()(const cl::sycl::device& Device) const override {
    const std::string DeviceName = Device.get_info<cl::sycl::info::device::name>();
    const std::string DeviceVendor = Device.get_info<cl::sycl::info::device::vendor>();
     if (DeviceName.find("SYCL") != std::string::npos)
       std::cout << "\tHost device: " << DeviceName << "\n";
     else
       std::cout << "\t" << DeviceVendor << ": " << DeviceName << "\n";
    return DeviceName.find(device) != std::string::npos;
   }
 };