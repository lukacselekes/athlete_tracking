#include "tracking_controller.hpp"

namespace tracking
{

TrackingController::TrackingController(const yolo::YoloDetector &f_yoloDetector, const int &f_detectorFrequency)
    : m_yoloDetector(f_yoloDetector), m_tracker(cv::TrackerMIL::create()), m_detectorFrequency(f_detectorFrequency)
{
}

void TrackingController::runTracking(const Mat &f_inputFrame)
{
    // Init detection struct
    std::optional<yolo::Detection> detection;

    // Run detector or tracker
    if (shouldDetectorRun())
    {
        runDetector(f_inputFrame, detection);

        // Init tracker whenever detector was running
        if (detection.has_value())
        {
            m_tracker->init(f_inputFrame, detection.value().box);
            m_trackerInitialized = true;
        }
    }
    else
    {
        runTracker(f_inputFrame, detection);
    }

    // If detection was successful, save it
    if (detection.has_value())
    {
        m_detections.push_back(detection.value());
        m_wasDetectedInCurrentFrame = true;
    }
    else
    {
        m_wasDetectedInCurrentFrame = false;
    }

    // Increase frame counter
    m_frameCounter++;
}

void TrackingController::runDetector(const Mat &f_inputFrame, std::optional<yolo::Detection> &f_detection)
{
    std::cout << "Running YOLO detector" << std::endl;

    // Run inference using yolo detector
    std::vector<yolo::Detection> output = m_yoloDetector.runInference(f_inputFrame);

    // If no person was detected, return
    if (output.empty())
    {
        m_wasDetectedInCurrentFrame = false;
        return;
    }

    assert(output.size() == 1 && "Only one person should be detected");

    // Save detection
    auto &detection       = output[0];
    detection.frameNumber = m_frameCounter;
    detection.type        = yolo::Detection::EType::Detected;

    f_detection = detection;
}

void TrackingController::runTracker(const Mat &f_inputFrame, std::optional<yolo::Detection> &f_detection)
{
    assert(m_trackerInitialized && "Tracker should be initialized");

    std::cout << "Running tracker" << std::endl;

    cv::Rect bbox = getLastDetection().box;
    if (m_tracker->update(f_inputFrame, bbox))
    {
        // If tracker was successful, build detection struct and set output
        yolo::Detection detection;
        detection.box         = bbox;
        detection.frameNumber = m_frameCounter;
        detection.color       = cv::Scalar(0, 255, 0);
        detection.className   = "person";
        detection.confidence  = 0.0;
        detection.type        = yolo::Detection::EType::Tracked;

        f_detection = detection;
    }
}

bool TrackingController::shouldDetectorRun()
{
    return (m_frameCounter % m_detectorFrequency == 0) || (!m_trackerInitialized);
}

yolo::Detection TrackingController::getLastDetection() const
{
    return m_detections.back();
}

void TrackingController::drawDetectionOnFrame(const Mat &f_inputFrame, const yolo::Detection &f_detection)
{
    const cv::Rect   &box   = f_detection.box;
    const cv::Scalar &color = f_detection.color;

    // Draw bounding box
    cv::rectangle(f_inputFrame, box, color, 2);

    std::string detectionType;
    if (f_detection.type == yolo::Detection::EType::Detected)
    {
        detectionType = "Detected";
    }
    else
    {
        detectionType = "Tracked";
    }

    // Draw label
    // std::string label = f_detection.className + ": " + std::to_string(f_detection.confidence).substr(0, 4);
    std::string label = f_detection.className + ": " + detectionType;

    cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
    cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
    cv::rectangle(f_inputFrame, textBox, color, cv::FILLED);
    cv::putText(f_inputFrame, label, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0),
                2, 0);
}

void TrackingController::drawTrajectoryOnFrame(const Mat &f_inputFrame) const
{
    // At least two detections are needed for drawing a trajectory
    if (m_detections.size() < 2)
    {
        return;
    }

    for (size_t i = 1; i < m_detections.size(); ++i)
    {
        // Get center point of two consecutive detections
        const cv::Point &p1 = m_detections[i - 1].box.tl() +
                              cv::Point(m_detections[i - 1].box.width / 2, m_detections[i - 1].box.height / 2);
        const cv::Point &p2 =
            m_detections[i].box.tl() + cv::Point(m_detections[i].box.width / 2, m_detections[i].box.height / 2);

        // Draw line between two centers of detections
        cv::line(f_inputFrame, p1, p2, m_detections[i].color, 3);

        // Draw center points
        cv::circle(f_inputFrame, p1, 3, m_detections[i - 1].color, cv::FILLED);
        cv::circle(f_inputFrame, p2, 3, m_detections[i].color, cv::FILLED);
    }
}

void TrackingController::saveDetectionsToFile(const std::string &f_filePath) const
{
    std::ofstream file(f_filePath);

    // Write detections to file
    for (const auto &detection : m_detections)
    {
        file << detection.frameNumber << " " << detection.box.x << " " << detection.box.y << " " << detection.box.width
             << " " << detection.box.height << std::endl;
    }

    std::cout << "Detections were saved to: " << f_filePath << std::endl;

    file.close();
}

bool TrackingController::wasDetectedInCurrentFrame() const
{
    return m_wasDetectedInCurrentFrame;
}

} // namespace tracking
