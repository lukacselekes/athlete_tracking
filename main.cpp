#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "tracking_controller.hpp"
#include "yolo_detector.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    std::string projectBasePath = "D:/Projects/athlete_tracking"; // Set base path

    bool runOnGPU = false;

    // Init yolo detector
    yolo::Inference yolo(projectBasePath + "/yolov8n.onnx", cv::Size(640, 640), runOnGPU);

    // Init video capture
    auto         video_name = "drill_1";
    auto         extension  = ".mp4";
    VideoCapture cap(projectBasePath + "/assets/" + video_name + extension);

    // Check if video opened successfully
    if (!cap.isOpened()) // check if we succeeded
        return -1;

    // Window for displaying
    namedWindow("Video", 1);

    const auto out_width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const auto out_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    const float scale = 0.8;

    VideoWriter out_video(projectBasePath + "/" + video_name + "_out" + extension,
                          VideoWriter::fourcc('m', 'p', '4', 'v'), 30, Size(scale * out_width, scale * out_height));

    for (;;)
    {
        Mat frame;

        // Read frame by frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        auto output_frame = tracking::detectPersonOnFrame(frame, yolo);

        // This is only for preview purposes
        cv::resize(output_frame, output_frame, cv::Size(out_width * scale, out_height * scale));

        // Display the resulting frame
        imshow("Video", output_frame);

        // Save the resulting frame
        out_video.write(output_frame);

        // Stop if key is pressed
        if (waitKey(30) >= 0)
            break;
    }

    cap.release();
    out_video.release();
    destroyAllWindows();

    return 0;
}
