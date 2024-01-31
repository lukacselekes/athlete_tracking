#include "tracking_controller.hpp"

namespace tracking
{
void runTracking(const Mat &f_input_frame, yolo::Inference &f_yolo_detector)
{
    // Mat l_detected_person = detectPersonOnFrame(f_input_frame, f_yolo_detector);
}

Mat detectPersonOnFrame(const Mat &f_input_frame, yolo::Inference &f_yolo_detector)
{
    // Run inference using yolo detector
    std::vector<yolo::Detection> output = f_yolo_detector.runInference(f_input_frame);

    int detections = output.size();
    std::cout << "Number of detections:" << detections << std::endl;

    // Make a copy of the input frame
    Mat output_frame = f_input_frame.clone();

    // Process detections
    for (int i = 0; i < detections; ++i)
    {
        yolo::Detection detection = output[i];

        if (detection.className == "person")
        {
            const cv::Rect   &box   = detection.box;
            const cv::Scalar &color = detection.color;

            // Draw bounding box
            cv::rectangle(output_frame, box, color, 2);

            // Draw label
            std::string label    = detection.className + ": " + std::to_string(detection.confidence).substr(0, 4);
            cv::Size    textSize = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect    textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
            cv::rectangle(output_frame, textBox, color, cv::FILLED);
            cv::putText(output_frame, label, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1,
                        cv::Scalar(0, 0, 0), 2, 0);
        }
    }

    return output_frame;
}

} // namespace tracking
