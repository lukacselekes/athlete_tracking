#include <opencv2/opencv.hpp>
#include <string>

namespace config
{

const std::string PROJECT_BASE_PATH = "D:/Projects/athlete_tracking";
const std::string YOLO_MODEL_PATH   = PROJECT_BASE_PATH + "/model/yolov8n.onnx";
const cv::Size    MODEL_INPUT_SHAPE(640, 640);
const bool        RUN_ON_GPU = false;

constexpr float OUTPUT_SCALE = 0.8F;
const int       VIDEO_FPS    = 30;

const std::string VIDEO_NAME   = "drill_1";
const std::string VIDEO_FORMAT = ".mp4";
const std::string VIDEO_PATH   = PROJECT_BASE_PATH + "/assets/" + VIDEO_NAME + VIDEO_FORMAT;

const std::string OUTPUT_DIR = PROJECT_BASE_PATH + "/output";

const bool DRAW_TRAJECTORY = true;

} // namespace config
