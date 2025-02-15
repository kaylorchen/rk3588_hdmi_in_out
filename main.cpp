#include <thread>

#include "framebuffer.h"
#include "hdmi_in.h"
#include "kaylordut/log/logger.h"
#include "opencv2/opencv.hpp"
#include "utils/tools.h"
#include "utils/yolo_threadpool.h"
#include "yaml-cpp/yaml.h"

double get_now() {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::duration<double>>(duration)
      .count();
}
int main(int argc, char **argv) {
  std::stringstream ss;
  for (int i = 0; i < argc; ++i) {
    ss << argv[i] << " ";
  }
  KAYLORDUT_LOG_INFO("Command: {}", ss.str());

  YAML::Node postprocess_config =
      YAML::LoadFile("../config/yolo_postprocess.yaml");
  std::vector<float> confidence_thresholds;
  std::vector<std::string> labels;
  for (const auto &kv : postprocess_config) {
    labels.push_back(kv.first.as<std::string>());
    confidence_thresholds.push_back(
        kv.second["confidence_threshold"].as<float>());
  }
  std::string model_path = "../model/yolo11n-seg.rknn";
  YoloThreadpool yolo_threadpool(model_path, confidence_thresholds, 6);

  auto info = cv::getBuildInformation();
  KAYLORDUT_LOG_INFO("Info: {}", info);
  auto config_node = YAML::LoadFile("../config/config.yaml");
  std::string device = config_node["hdmi_in"].as<std::string>();
  HdmiIn hdmi_in(device);
  std::string fb = config_node["framebuffer"].as<std::string>();
  FrameBuffer frame_buffer(fb);
  std::vector<cv::Mat> input(1);
  std::shared_ptr<YoloThreadpool::YoloInferenceResult> result = nullptr;
  while (true) {
    KAYLORDUT_TIME_COST_DEBUG("read frame",
                              input.at(0) = hdmi_in.get_next_frame());
    if (!input.at(0).empty()) {
      yolo_threadpool.AddInferenceTask(input, get_now(), true);
    }
    result = yolo_threadpool.GetInferenceResult();
    if (result != nullptr) {
      cv::Mat res_image;
      KAYLORDUT_TIME_COST_DEBUG(
          "GetResultImage", res_image = GetImageResult(
                                result->original_image.at(0),
                                yolo_threadpool.get_model_input_side_length(),
                                result->results, labels, false));
      KAYLORDUT_TIME_COST_DEBUG("write frame",
                                frame_buffer.WriteFrameBuffer(res_image));
    } else {
    }
  }
  // cv::destroyAllWindows();
  return 0;
}
