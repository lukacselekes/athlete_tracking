cmake_minimum_required(VERSION 3.5)
project(athlete_tracking VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})


add_executable(athlete_tracking 
sources/main.cpp

sources/yolo_detector.hpp
sources/yolo_detector.cpp

sources/tracking_controller.hpp
sources/tracking_controller.cpp

sources/tracking_config.hpp)

target_link_libraries(athlete_tracking ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
