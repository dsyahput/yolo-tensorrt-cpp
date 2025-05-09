#pragma once

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <memory>
#include "deploy/model.hpp"
#include "deploy/option.hpp"
#include "deploy/result.hpp"

struct Config {
    std::string engine_path = "../models/yolo11n.engine";
    std::string label_path = "../models/labels.txt";
    float confidence_threshold = 0.5f;
    int camera_id = 0;
    bool use_video = false;
    std::string video_path;
    bool save_output = false;
    int camera_width = 640;
    int camera_height = 480;
};

class YoloDetector {
public:
    explicit YoloDetector(const Config& config);
    void run();

private:
    Config config_;
    std::unique_ptr<deploy::DetectModel> model_;
    std::vector<std::string> labels_;
    int saved_frame_count_ = 0;
    cv::VideoWriter video_writer_;
    bool video_writer_initialized_ = false;

    void loadLabels();
    void loadModel();
    void ensureOutputDirExists();
    void visualize(cv::Mat& image, const deploy::DetectRes& result);
    cv::Scalar getColor(int class_id, int num_classes);
    void initializeVideoWriter(int width, int height, double fps);
};
