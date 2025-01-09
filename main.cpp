#include <thread>

#include "framebuffer.h"
#include "hdmi_in.h"
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
  std::string device = "/dev/video11";
  HdmiIn hdmi_in(device);
  std::string fb = "/dev/fb0";
  FrameBuffer frame_buffer(fb);
  while(true){
    auto frame = hdmi_in.get_next_frame();
    if (!frame.empty()) {
      // cv::imshow(device, frame);
      // if (cv::waitKey(1) == 'q') {
      //   break;
      // }
      frame_buffer.WriteFrameBuffer(frame);
    }
  }
  // cv::destroyAllWindows();
  return 0;
}
