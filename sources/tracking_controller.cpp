#include "tracking_controller.hpp"

namespace tracking
{

TrackingController::TrackingController(const yolo::YoloDetector &f_yoloDetector) : m_yoloDetector(f_yoloDetector)
{
}

void TrackingController::runTracking(const Mat &f_inputFrame)
{
    // Increase frame counter
    m_frameCounter++;

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
    auto &detection = output[0];
    // Set frame number
    detection.frameNumber = m_frameCounter;

    m_detections.push_back(detection);
    m_wasDetectedInCurrentFrame = true;
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

    // Draw label
    std::string label    = f_detection.className + ": " + std::to_string(f_detection.confidence).substr(0, 4);
    cv::Size    textSize = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
    cv::Rect    textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
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
