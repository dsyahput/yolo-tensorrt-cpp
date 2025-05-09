#include "YoloDetector.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <sstream>

namespace fs = std::filesystem;

YoloDetector::YoloDetector(const Config& config) : config_(config) {
    loadLabels();
    loadModel();
    if (config_.save_output) {
        ensureOutputDirExists();
    }
}

void YoloDetector::loadLabels() {
    std::ifstream file(config_.label_path);
    if (!file.is_open()) throw std::runtime_error("Failed to open labels file");

    std::string label;
    while (std::getline(file, label)) {
        if (!label.empty()) labels_.emplace_back(label);
    }

    if (labels_.empty()) throw std::runtime_error("No labels found in file");
}

void YoloDetector::loadModel() {
    if (!fs::exists(config_.engine_path)) {
        throw std::runtime_error("Engine file not found: " + config_.engine_path);
    }

    deploy::InferOption option;
    option.enableSwapRB();
    model_ = std::make_unique<deploy::DetectModel>(config_.engine_path, option);
}

void YoloDetector::ensureOutputDirExists() {
    std::string dir = "../output";
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
        std::cout << "Created output directory: " << dir << "\n";
    }
}

cv::Scalar YoloDetector::getColor(int class_id, int num_classes) {
    float hue = static_cast<float>(class_id) / num_classes * 180.0f;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 200, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr.at<cv::Vec3b>(0, 0);
}

void YoloDetector::visualize(cv::Mat& image, const deploy::DetectRes& result) {
    for (size_t i = 0; i < result.num; ++i) {
        float score = result.scores[i];
        if (score < config_.confidence_threshold) continue;

        const auto& box = result.boxes[i];
        int cls = result.classes[i];
        if (cls < 0 || cls >= static_cast<int>(labels_.size())) continue;

        std::string label = labels_[cls] + " " + cv::format("%.2f", score);
        cv::Scalar color = getColor(cls, labels_.size());

        int left = std::max(0, static_cast<int>(box.left));
        int top = std::max(0, static_cast<int>(box.top));
        int right = std::min(image.cols - 1, static_cast<int>(box.right));
        int bottom = std::min(image.rows - 1, static_cast<int>(box.bottom));

        int base_line;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base_line);

        cv::rectangle(image, {left, top}, {right, bottom}, color, 2);
        cv::rectangle(image, {left, top - label_size.height - 5}, {left + label_size.width, top}, color, -1);
        cv::putText(image, label, {left, top - 5}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    }
}

void YoloDetector::initializeVideoWriter(int width, int height, double fps) {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << "../output/output_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".avi";
    std::string out_path = oss.str();

    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    video_writer_.open(out_path, codec, fps, cv::Size(width, height));

    if (!video_writer_.isOpened()) {
        throw std::runtime_error("Failed to open video writer at " + out_path);
    }

    std::cout << "Saving full video to: " << out_path << "\n";
    video_writer_initialized_ = true;
}

void YoloDetector::run() {
    cv::VideoCapture cap;

    if (config_.use_video) {
        std::cout << "Opening video: " << config_.video_path << "\n";
        cap.open(config_.video_path);
    } else {
        std::cout << "Opening camera: " << config_.camera_id << "\n";
        cap.open(config_.camera_id);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, config_.camera_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, config_.camera_height);
    }

    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open camera or video stream.");
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 1.0 || fps > 120.0) fps = 30.0;

    std::cout << "Press 'ESC' to exit.\n";

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Failed to read frame.\n";
            break;
        }

        if (config_.save_output && !video_writer_initialized_) {
            initializeVideoWriter(frame.cols, frame.rows, fps);
        }

        deploy::Image input(frame.data, frame.cols, frame.rows);
        auto result = model_->predict(input);

        cv::Mat display = frame.clone();
        visualize(display, result);
        cv::imshow("YOLO Detection", display);

        if (config_.save_output && video_writer_initialized_) {
            video_writer_.write(display);
        }

        int key = cv::waitKey(1);
        if (key == 27) break;  // ESC
        else if ((key == 's' || key == 'S') && config_.save_output) {
            std::time_t t = std::time(nullptr);
            std::tm* now = std::localtime(&t);
            
            char filename[100];
            std::strftime(filename, sizeof(filename), "../output/detection_%Y%m%d_%H%M%S.jpg", now);

            cv::imwrite(filename, display);
            std::cout << "Saved frame to " << filename << "\n";
        }
    }

    if (video_writer_initialized_) {
        video_writer_.release();
    }

    std::cout << "Detection finished.\n";
}
