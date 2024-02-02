#include "tracking_controller.hpp"

namespace tracking
{

TrackingController::TrackingController(const yolo::Inference &f_yolo_detector)
    : m_yolo_detector(f_yolo_detector)
{
}

void TrackingController::runTracking(const Mat &f_input_frame)
{
    // Run inference using yolo detector
    std::vector<yolo::Detection> output = m_yolo_detector.runInference(f_input_frame);

    assert(output.size() == 1 && "Only one person should be detected");

    m_detections.push_back(output[0]);
}

yolo::Detection TrackingController::getLastDetection() const
{
    return m_detections.back();
}

void TrackingController::drawDetectionOnFrame(const Mat &f_input_frame, yolo::Detection &f_detection)
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

} // namespace tracking
