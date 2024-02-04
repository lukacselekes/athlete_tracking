#ifndef YOLO_DETECTOR_HPP
#define YOLO_DETECTOR_HPP

// Cpp native
#include <random>
#include <string>
#include <vector>

// OpenCV / DNN / Inference
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace yolo
{

struct Detection
{
    enum EType
    {
        Detected = 0,
        Tracked
    };

    int         class_id{0};
    std::string className{};
    float       confidence{0.0};
    cv::Scalar  color{};
    cv::Rect    box{};
    int         frameNumber{0};
    EType       type{Detected};
};

using DetectionVector = std::vector<Detection>;

class YoloDetector
{
  public:
    YoloDetector(const std::string &onnxModelPath, const cv::Size &modelInputShape = {640, 640},
                 const bool &runWithCuda = true);
    DetectionVector runInference(const cv::Mat &input);

  private:
    void    loadOnnxNetwork();
    cv::Mat formatToSquare(const cv::Mat &source);

    std::string modelPath{};
    bool        cudaEnabled{};

    std::vector<std::string> classes{"person"};

    cv::Size2f modelShape{};

    float modelConfidenceThreshold{0.25};
    float modelScoreThreshold{0.45};
    float modelNMSThreshold{0.50};

    bool letterBoxForSquare = true;

    cv::dnn::Net net;
};

} // namespace yolo

#endif // YOLO_DETECTOR_HPP
