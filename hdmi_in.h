//
// Created by kaylor on 1/8/25.
//

#ifndef HDMI_IN_H
#define HDMI_IN_H

#include "opencv2/opencv.hpp"

class HdmiIn {
 public:
  HdmiIn(std::string device_name);
  ~HdmiIn();
  cv::Mat get_next_frame();

 private:
  cv::VideoCapture capture_;
  std::string device_name_;
};

#endif  // HDMI_IN_H
