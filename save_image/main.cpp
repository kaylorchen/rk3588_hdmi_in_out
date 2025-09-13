//
// Created by kaylor on 9/13/25.
//

#include <hdmi_in.h>
#include "kaylordut/log/logger.h"
#include "opencv2/opencv.hpp"
#include "sstream"

int main(int argc, char **argv) {

  std::stringstream cmd;
  for (int i = 0; i < argc; ++i) {
    cmd << argv[i] << " ";
  }
  KAYLORDUT_LOG_INFO("Command: {}", cmd.str());
  std::string device_name = "/dev/video0";
  HdmiIn device(device_name);
  int count = 10;
  cv::Mat frame;
  while (count--) {
    frame = device.get_next_frame();
  }
  cv::imshow("image", frame);
  cv::waitKey(500);
  // 获取当前时间
  auto now = std::chrono::system_clock::now();
  auto now_time = std::chrono::system_clock::to_time_t(now);

  // 格式化为文件名（YYYY-MM-DD_HH-MM-SS）
  std::stringstream ss;
  ss << std::put_time(std::localtime(&now_time), "%Y-%m-%d_%H-%M-%S");
  std::string timestamp = ss.str();

  // 构造文件名
  std::string filename = "image_" + timestamp + ".png";

  // 保存图像
  bool saved = cv::imwrite(filename, frame);

  if (saved) {
    KAYLORDUT_LOG_INFO("Image saved as: {}", filename);
  } else {
    KAYLORDUT_LOG_ERROR("Failed to save image!");
  }
  cv::destroyAllWindows();
  return 0;
}