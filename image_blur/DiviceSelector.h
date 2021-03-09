#pragma once
#include <CL/sycl.hpp>


class DeviceSelector : public cl::sycl::device_selector {
  std::string my_device;
public:
  DeviceSelector(std::string _my_device) : my_device(_my_device) {}

  int operator()(const cl::sycl::device& Device) const override {
    const std::string DeviceName = Device.get_info<cl::sycl::info::device::name>();
    return DeviceName.find(my_device) != std::string::npos;
   }
 };