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
    TrackingController(const yolo::YoloDetector &f_yoloDetector);

    void runTracking(const Mat &f_inputFrame);

    yolo::Detection getLastDetection() const;

    static void drawDetectionOnFrame(const Mat &f_inputFrame, const yolo::Detection &f_detection);

    void drawTrajectoryOnFrame(const Mat &f_inputFrame) const;

    void saveDetectionsToFile(const std::string &f_filePath) const;

    bool wasDetectedInCurrentFrame() const;

  private:
    yolo::YoloDetector       m_yoloDetector;
    yolo::DetectionVector m_detections;

    bool m_wasDetectedInCurrentFrame = false;
    int  m_frameCounter              = 0;
};

} // namespace tracking

#endif // TRACKING_CONTROLLER_HPP
