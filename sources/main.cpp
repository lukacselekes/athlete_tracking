#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "tracking_config.hpp"
#include "tracking_controller.hpp"
#include "yolo_detector.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    // Init yolo detector
    yolo::YoloDetector yolo(config::YOLO_MODEL_PATH, config::MODEL_INPUT_SHAPE, config::RUN_ON_GPU);

    // Init video capture
    VideoCapture cap(config::VIDEO_PATH);

    // Check if video opened successfully
    if (!cap.isOpened()) // check if we succeeded
    {
        cout << "Could not open video: " << config::VIDEO_PATH << endl;
        return -1;
    }

    // Window for displaying
    namedWindow("Video", 1);

    // Get size of the frames
    const auto out_width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const auto out_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Create output directory if it does not exist
    if (!std::filesystem::exists(config::OUTPUT_DIR))
    {
        std::filesystem::create_directory(config::OUTPUT_DIR);
    }

    // Init video writer
    const string outputVideoPath = config::OUTPUT_DIR + "/" + config::VIDEO_NAME + "_out" + config::VIDEO_FORMAT;
    VideoWriter  outVideo(outputVideoPath, VideoWriter::fourcc('m', 'p', '4', 'v'), config::VIDEO_FPS,
                          Size(config::OUTPUT_SCALE * out_width, config::OUTPUT_SCALE * out_height));

    // Init tracking controller
    tracking::TrackingController trackingController(yolo);

    for (;;)
    {
        Mat frame;

        // Read frame by frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
        {
            cout << "Frame is empty" << endl;
            break;
        }

        // Start timer
        double timer = (double)getTickCount();

        // Run tracking
        trackingController.runTracking(frame);

        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);

        // If no detection was made in the current frame, continue to the next frame
        if (!trackingController.wasDetectedInCurrentFrame())
        {
            continue;
        }

        // Get last detection to draw on the frame
        auto lastDetection = trackingController.getLastDetection();

        // Draw detection on the frame
        trackingController.drawDetectionOnFrame(frame, lastDetection);

        // Draw trajectory on the frame
        if (config::DRAW_TRAJECTORY)
        {
            trackingController.drawTrajectoryOnFrame(frame);
        }

        // This is only for preview purposes
        cv::resize(frame, frame, cv::Size(out_width * config::OUTPUT_SCALE, out_height * config::OUTPUT_SCALE));

        // Display FPS on frame
        putText(frame, "FPS: " + to_string(int(fps)), Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 2);

        // Display the resulting frame
        imshow("Video", frame);

        // Save the resulting frame
        outVideo.write(frame);

        // Stop if key is pressed
        if (waitKey(30) >= 0)
            break;
    }

    cap.release();
    outVideo.release();
    destroyAllWindows();

    // Print some messages
    cout << "Video processing finished" << endl;
    cout << "Output video was saved to: " << outputVideoPath << endl;

    // Save detections to file
    const string detectionsFilePath = config::OUTPUT_DIR + "/" + config::VIDEO_NAME + "_detections.txt";
    trackingController.saveDetectionsToFile(detectionsFilePath);

    return 0;
}
