#include "kaylordut/log/logger.h"
#include "opencv2/opencv.hpp"
#include "framebuffer.h"
int main(int argc, char *argv[]){
  std::string video_filename = std::string(argv[1]);
  KAYLORDUT_LOG_INFO("video file is {}", video_filename);
  cv::VideoCapture video_capture(video_filename);
  if (!video_capture.isOpened()) {
    KAYLORDUT_LOG_ERROR("Failed to open video file {}", video_filename);
    return -1;
  }
  auto fps = video_capture.get(cv::CAP_PROP_FPS);
  KAYLORDUT_LOG_INFO("fps is {}", fps);
  auto width = video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
  auto height = video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  KAYLORDUT_LOG_INFO("resolution is {}x{} ", width, height);
  auto total_frames = video_capture.get(cv::CAP_PROP_FRAME_COUNT);
  KAYLORDUT_LOG_INFO("total frames is {}", total_frames);
  FrameBuffer frame_buffer("/dev/fb0");
  cv::Mat frame;
  while (true) {
    video_capture >> frame;
    if (frame.empty()) {
      KAYLORDUT_LOG_INFO("End of Video");
      break;
    }
    frame_buffer.WriteFrameBuffer(frame);
  }
  video_capture.release();
  return 0;
}