//
// Created by kaylor on 1/8/25.
//

#include "hdmi_in.h"

#include "kaylordut/log/logger.h"

HdmiIn::HdmiIn(std::string device_name) {
  device_name_ = device_name;
  capture_ = cv::VideoCapture(
      "v4l2src device=" + device_name_ +
          " ! videoconvert ! video/x-raw, format=BGR ! appsink",
      cv::CAP_GSTREAMER);
  if (!capture_.isOpened()) {
    KAYLORDUT_LOG_ERROR("cannot open {}", device_name_);
    exit(EXIT_FAILURE);
  }
}

HdmiIn::~HdmiIn() {
  if (capture_.isOpened()) {
    capture_.release();
  }
}

cv::Mat HdmiIn::get_next_frame() {
  cv::Mat frame;
  capture_ >> frame;
  if (frame.empty()) {
    KAYLORDUT_LOG_WARN("cannot read a new frame");
    // return cv::Mat();
  }
  return frame;
}
