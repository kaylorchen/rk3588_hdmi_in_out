#include <thread>
#include "kaylordut/log/logger.h"
#include "opencv2/opencv.hpp"
int main(int argc, char **argv) {
  std::stringstream ss;
  for (int i = 0; i < argc; ++i) {
    ss << argv[i] << " ";
  }
  KAYLORDUT_LOG_INFO("Command: {}", ss.str());
  auto info = cv::getBuildInformation();
  KAYLORDUT_LOG_INFO("Info: {}", info);
  std::string device = "/dev/video0";
//  cv::VideoCapture cap(0, cv::CAP_GSTREAMER);
  cv::VideoCapture cap("v4l2src device=" + device + " ! videoconvert ! video/x-raw, format=BGR ! appsink", cv::CAP_GSTREAMER);
  if (!cap.isOpened()){
    KAYLORDUT_LOG_ERROR("can't open {}", device);
    exit(EXIT_FAILURE);
  }
  cv::Mat frame;
  while(true){
    cap >> frame;
    if (frame.empty()){
      KAYLORDUT_LOG_ERROR("can't read frame");
      break;
    }
    cv::imshow(device, frame);
    if (cv::waitKey(1) == 'q'){
      break;
    }
  }
  cap.release();
  cv::destroyAllWindows();
  return 0;
}
