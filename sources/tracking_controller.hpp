#ifndef TRACKING_CONTROLLER_HPP
#define TRACKING_CONTROLLER_HPP

#include <fstream>
#include <opencv2/opencv.hpp>

#include "yolo_detector.hpp"

using namespace cv;

namespace tracking
{

class TrackingController
{
  public:
    TrackingController(const yolo::Inference &f_yolo_detector);

    void runTracking(const Mat &f_input_frame);

    yolo::Detection getLastDetection() const;

    static void drawDetectionOnFrame(const Mat &f_input_frame, const yolo::Detection &f_detection);

    void drawTrajectoryOnFrame(const Mat &f_input_frame) const;

    void saveDetectionsToFile(const std::string &f_filePath) const;

    bool wasDetectedInCurrentFrame() const;

  private:
    yolo::Inference       m_yolo_detector;
    yolo::DetectionVector m_detections;

    bool m_wasDetectedInCurrentFrame = false;
    int  m_frameCounter              = 0;
};

} // namespace tracking

#endif // TRACKING_CONTROLLER_HPP
