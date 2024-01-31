#ifndef TRACKING_CONTROLLER_HPP
#define TRACKING_CONTROLLER_HPP

#include <opencv2/opencv.hpp>
#include "yolo_detector.hpp"

using namespace cv;

namespace tracking
{
    void runTracking(const Mat& f_input_frame, yolo::Inference& f_yolo_detector);

    Mat detectPersonOnFrame(const Mat& f_input_frame, yolo::Inference& f_yolo_detector);

} // namespace tracking


#endif // TRACKING_CONTROLLER_HPP
