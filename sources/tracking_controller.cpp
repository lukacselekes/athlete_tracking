#include "tracking_controller.hpp"

namespace tracking
{

TrackingController::TrackingController(const yolo::Inference &f_yolo_detector) : m_yolo_detector(f_yolo_detector)
{
}

void TrackingController::runTracking(const Mat &f_input_frame)
{
    // Run inference using yolo detector
    std::vector<yolo::Detection> output = m_yolo_detector.runInference(f_input_frame);

    if (output.empty())
    {
        m_wasDetectedInCurrentFrame = false;
        return;
    }

    assert(output.size() == 1 && "Only one person should be detected");

    m_detections.push_back(output[0]);
    m_wasDetectedInCurrentFrame = true;
}

yolo::Detection TrackingController::getLastDetection() const
{
    return m_detections.back();
}

void TrackingController::drawDetectionOnFrame(const Mat &f_input_frame, const yolo::Detection &f_detection)
{
    const cv::Rect   &box   = f_detection.box;
    const cv::Scalar &color = f_detection.color;

    // Draw bounding box
    cv::rectangle(f_input_frame, box, color, 2);

    // Draw label
    std::string label    = f_detection.className + ": " + std::to_string(f_detection.confidence).substr(0, 4);
    cv::Size    textSize = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
    cv::Rect    textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
    cv::rectangle(f_input_frame, textBox, color, cv::FILLED);
    cv::putText(f_input_frame, label, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0),
                2, 0);
}

void TrackingController::drawTrajectoryOnFrame(const Mat &f_input_frame) const
{
    if (m_detections.size() < 2)
    {
        return;
    }

    for (size_t i = 1; i < m_detections.size(); ++i)
    {
        const cv::Point &p1 = m_detections[i - 1].box.tl() +
                              cv::Point(m_detections[i - 1].box.width / 2, m_detections[i - 1].box.height / 2);
        const cv::Point &p2 =
            m_detections[i].box.tl() + cv::Point(m_detections[i].box.width / 2, m_detections[i].box.height / 2);

        cv::line(f_input_frame, p1, p2, m_detections[i].color, 3);

        cv::circle(f_input_frame, p1, 3, m_detections[i - 1].color, cv::FILLED);
        cv::circle(f_input_frame, p2, 3, m_detections[i].color, cv::FILLED);
    }
}

void TrackingController::saveDetectionsToFile(const std::string &f_filePath) const
{
    std::ofstream file(f_filePath);

    for (const auto &detection : m_detections)
    {
        file << detection.box.x << " " << detection.box.y << " " << detection.box.width << " " << detection.box.height
             << std::endl;
    }

    std::cout << "Detections were saved to: " << f_filePath << std::endl;

    file.close();
}

bool TrackingController::wasDetectedInCurrentFrame() const
{
    return m_wasDetectedInCurrentFrame;
}

} // namespace tracking
