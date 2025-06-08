cmake_minimum_required(VERSION 3.21)
project(rk3588_hdmi_in)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
find_package(kaylordut REQUIRED)

file(GLOB SRC "./*.cpp")

add_executable(rk3588_hdmi_in ${SRC})
target_link_libraries(rk3588_hdmi_in PRIVATE
        ${OpenCV_LIBS}
        ${kaylordut_LIBS}
)
target_include_directories(rk3588_hdmi_in PRIVATE ${OpenCV_INCLUDE_DIRS} )

add_executable(update_edid update_edid.cpp)
target_link_libraries(update_edid ${kaylordut_LIBS})
