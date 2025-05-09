#include "YoloDetector.hpp"
#include <iostream>
#include <memory>

Config parse_arguments(int argc, char* argv[]) {
    Config config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--engine" && i + 1 < argc) {
            config.engine_path = argv[++i];
        } else if (arg == "--labels" && i + 1 < argc) {
            config.label_path = argv[++i];
        } else if (arg == "--threshold" && i + 1 < argc) {
            config.confidence_threshold = std::stof(argv[++i]);
        } else if (arg == "--camera" && i + 1 < argc) {
            config.camera_id = std::stoi(argv[++i]);
        } else if (arg == "--video" && i + 1 < argc) {
            config.use_video = true;
            config.video_path = argv[++i];
        } else if (arg == "--save") {
            config.save_output = true;
            std::cout << "Press 'S' to save current frame.\n";
        } else if (arg == "--resolution" && i + 2 < argc) {
            config.camera_width = std::stoi(argv[++i]);
            config.camera_height = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n"
                      << "Options:\n"
                      << "  --engine PATH       Path to the TensorRT model engine file.\n"
                      << "  --labels PATH       Path to the labels file.\n"
                      << "  --threshold VALUE   Confidence threshold for object detection.\n"
                      << "  --camera ID         Camera device ID (default: 0).\n"
                      << "  --video PATH        Use video file instead of camera.\n"
                      << "  --save              Save inference video to ../output/. Press 'S' to save current frame as an image in ../output/.\n"
                      << "  --resolution W H    Set camera resolution (default: 640x480).\n"
                      << "  --help              Show this help message.\n";
            exit(0);
        }
    }

    return config;
}


int main(int argc, char* argv[]) {
    try {
        Config config = parse_arguments(argc, argv);

        std::unique_ptr<YoloDetector> detector = std::make_unique<YoloDetector>(config);
        detector->run(); 

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
