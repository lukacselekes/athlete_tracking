#include <iostream>
#include <vector>
// #include <getopt.h>

#include <opencv2/opencv.hpp>

#include "yolo_detector.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    std::string projectBasePath = "D:/Projects/athlete_tracking"; // Set  base path

    bool runOnGPU = false;

    // Init yolo detector
    yolo::Inference yolo(projectBasePath + "/yolov8n.onnx", cv::Size(640, 640), runOnGPU);

    // Init video capture
    VideoCapture cap(projectBasePath + "/assets/drill_1.mp4");
    
    // Check if video opened successfully
    if (!cap.isOpened()) // check if we succeeded
        return -1;

    namedWindow("Video", 1);

    for (;;)
    {
        Mat frame;

        // Read frame by frame
        cap >> frame; 

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        // Mat resized;
        // resize(frame, frame, Size(640, 480), INTER_LINEAR);

        // Run inference using yolo detector
        std::vector<yolo::Detection> output = yolo.runInference(frame);

        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

        // Process detections
        for (int i = 0; i < detections; ++i)
        {
            yolo::Detection detection = output[i];

            cv::Rect   box   = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(frame, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size    textSize    = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect    textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1,
                        cv::Scalar(0, 0, 0), 2, 0);
        }

        // This is only for preview purposes
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols * scale, frame.rows * scale));

        imshow("Video", frame);
        if (waitKey(30) >= 0)
            break;
    }

    return 0;
}
