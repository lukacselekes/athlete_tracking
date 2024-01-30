cmake_minimum_required(VERSION 3.5)
project(athlete_tracking VERSION 0.1 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})


add_executable(athlete_tracking 
main.cpp

yolo_detector.hpp
yolo_detector.cpp)

target_link_libraries(athlete_tracking ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
