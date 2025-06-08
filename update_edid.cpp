#include <linux/videodev2.h>

#include "fcntl.h"
#include "fstream"
#include "kaylordut/log/logger.h"
#include "sstream"
#include "sys/ioctl.h"
#include "sys/stat.h"
#include "unistd.h"

static uint8_t edid_data[256] = {0};

void UpdateEdid(char *device, uint8_t *edid_data, uint8_t block_num) {
  struct stat st;
  KAYLORDUT_LOG_INFO("Update edid");
  if (-1 == stat(device, &st)) {
    throw strerror(errno);
  }
  if (!S_ISCHR(st.st_mode)) {
    throw strerror(errno);
  }
  int fd = open(device, O_RDWR | O_NONBLOCK, 0);
  if (-1 == fd) {
    throw strerror(errno);
  }
  struct v4l2_edid write_edid = {0};
  write_edid.pad = 0;
  write_edid.blocks = 2;
  write_edid.edid = edid_data;
  write_edid.start_block = 0;
  auto ret = ioctl(fd, VIDIOC_S_EDID, &write_edid);
  if (-1 == ret) {
    KAYLORDUT_LOG_ERROR("VIDIOC_G_EDID failed, {}",
                        std::string(strerror(errno)));
    exit(EXIT_FAILURE);
  }

  struct v4l2_edid read_edid = {0};
  read_edid.pad = 0;
  read_edid.start_block = 0;
  read_edid.blocks = 2;
  uint8_t tmp[256] = {0};
  read_edid.edid = tmp;
  ret = ioctl(fd, VIDIOC_G_EDID, &read_edid);
  if (-1 == ret) {
    KAYLORDUT_LOG_ERROR("VIDIOC_G_EDID failed, {}",
                        std::string(strerror(errno)));
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < 256; i++) {
    printf("%02X ", tmp[i]);
  }
  printf("\n");
  close(fd);
}

int main(int argc, char *argv[]) {
  std::stringstream ss;
  for (int i = 1; i < argc; ++i) {
    ss << argv[i] << " ";
  }
  KAYLORDUT_LOG_INFO("Command: {}", ss.str());
  if (argc > 2) {
    KAYLORDUT_LOG_INFO("Read edid from {}", argv[2]);
    std::ifstream file(argv[2], std::ios::binary);
    if (!file) {
      KAYLORDUT_LOG_ERROR("Failed to open file {}", argv[2]);
      return 1;
    }
    file.read(reinterpret_cast<char *>(&edid_data[0]), 256);
    if (!file) {
      KAYLORDUT_LOG_ERROR("Failed to read file {}", argv[2]);
      return 1;
    }
    file.close();
  } else {
    KAYLORDUT_LOG_WARN("argc < 3");
  }
  UpdateEdid(argv[1], edid_data, 0);
  return 0;
}