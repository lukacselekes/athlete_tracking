#ifndef TRACKING_CONTROLLER_HPP
#define TRACKING_CONTROLLER_HPP

#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <optional>

#include "yolo_detector.hpp"

using namespace cv;

namespace tracking
{

class TrackingController
{
  public:
    TrackingController(const yolo::YoloDetector &f_yoloDetector, const int &f_detectorFrequency = 10);

    void runTracking(const Mat &f_inputFrame);

    yolo::Detection getLastDetection() const;

    static void drawDetectionOnFrame(const Mat &f_inputFrame, const yolo::Detection &f_detection);

    void drawTrajectoryOnFrame(const Mat &f_inputFrame) const;

    void saveDetectionsToFile(const std::string &f_filePath) const;

    bool wasDetectedInCurrentFrame() const;

  private:
    void runDetector(const Mat &f_inputFrame, std::optional<yolo::Detection> &f_detection);

    void runTracker(const Mat &f_inputFrame, std::optional<yolo::Detection> &f_detection);

    bool shouldDetectorRun();

    yolo::YoloDetector    m_yoloDetector;
    yolo::DetectionVector m_detections;
    cv::Ptr<cv::Tracker>  m_tracker;

    int m_detectorFrequency;

    bool m_trackerInitialized        = false;
    bool m_wasDetectedInCurrentFrame = false;
    int  m_frameCounter              = 0;
};

} // namespace tracking

#endif // TRACKING_CONTROLLER_HPP
