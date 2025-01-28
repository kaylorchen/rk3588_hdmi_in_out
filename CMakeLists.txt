cmake_minimum_required(VERSION 3.21)
project(rk3588_hdmi_in)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
find_package(kaylordut REQUIRED)
find_package(ai_instance_rk3588 REQUIRED)
find_package(bytetrack REQUIRED)
find_package(yaml-cpp REQUIRED)
include_directories(./)

file(GLOB SRC "./*.cpp" "utils/*.cpp" "image_process/yolo/*.cpp")

add_executable(rk3588_hdmi_in ${SRC})
target_link_libraries(rk3588_hdmi_in PRIVATE
        ${OpenCV_LIBS}
        ${kaylordut_LIBS}
        ${ai_instance_rk3588_LIBS}
        ${bytetrack_LIBS}
        ${YAML_CPP_LIBRARIES}
)
target_include_directories(rk3588_hdmi_in PRIVATE ${OpenCV_INCLUDE_DIRS} )
