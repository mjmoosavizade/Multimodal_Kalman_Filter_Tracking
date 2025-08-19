#include "data_loader.h"
#include "perception_module.h"
#include "fusion_module.h"
#include "visualization.h"
#include <iostream>
#include <chrono>
#include <thread>

using namespace tracking;

int main(int argc, char** argv) {
    std::cout << "=== Multimodal Kalman Filter Tracking System ===" << std::endl;
    
    // Parse command line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <kitti_dataset_path> [model_path]" << std::endl;
        std::cerr << "Example: " << argv[0] << " /path/to/kitti/sequence_00" << std::endl;
        return -1;
    }
    
    std::string dataset_path = argv[1];
    std::string model_path = (argc > 2) ? argv[2] : "";
    
    // Initialize components
    std::cout << "Initializing components..." << std::endl;
    
    // 1. Data Loader
    KITTIDataLoader data_loader(dataset_path);
    if (!data_loader.initialize()) {
        std::cerr << "Failed to initialize data loader" << std::endl;
        return -1;
    }
    std::cout << "✓ Data loader initialized with " << data_loader.getNumFrames() << " frames" << std::endl;
    
    // 2. Perception Module
    PerceptionModule perception;
    if (!model_path.empty()) {
        if (!perception.initialize(model_path)) {
            std::cerr << "Failed to initialize perception module with model: " << model_path << std::endl;
            std::cout << "Continuing with LiDAR-only detection..." << std::endl;
        } else {
            std::cout << "✓ Perception module initialized with camera model" << std::endl;
        }
    } else {
        std::cout << "✓ Perception module initialized (LiDAR-only mode)" << std::endl;
    }
    
    // 3. Fusion Module
    FusionModule fusion;
    fusion.initialize(50, 5.0, 5); // max_tracks, association_threshold, max_consecutive_misses
    std::cout << "✓ Fusion module initialized" << std::endl;
    
    // 4. Visualizer
    Visualizer visualizer;
    if (!visualizer.initialize("Multimodal Tracking", true)) {
        std::cerr << "Failed to initialize visualizer" << std::endl;
        return -1;
    }
    std::cout << "✓ Visualizer initialized" << std::endl;
    
    // 5. Performance Profiler
    PerformanceProfiler profiler;
    
    std::cout << "\nStarting tracking loop..." << std::endl;
    std::cout << "Press 'q' to quit, 's' to save current frame, 'p' to pause" << std::endl;
    
    // Main processing loop
    bool paused = false;
    int frame_index = 0;
    int max_frames = std::min(data_loader.getNumFrames(), 1000); // Limit for demo
    
    while (frame_index < max_frames) {
        profiler.startTimer("total_frame");
        
        std::cout << "\rProcessing frame " << frame_index + 1 << "/" << max_frames << std::flush;
        
        // Load sensor data
        profiler.startTimer("data_loading");
        cv::Mat image = data_loader.loadCameraImage(0, frame_index); // Left camera
        auto point_cloud = data_loader.loadLiDARPointCloud(frame_index);
        IMUData imu_data = data_loader.loadIMUData(frame_index);
        double timestamp = data_loader.getTimestamp(frame_index);
        profiler.stopTimer("data_loading");
        
        if (image.empty() && (!point_cloud || point_cloud->empty())) {
            std::cerr << "\nSkipping frame " << frame_index << " - no valid data" << std::endl;
            frame_index++;
            continue;
        }
        
        // Get camera calibration
        Eigen::Matrix3d camera_matrix = data_loader.getCameraMatrix(0);
        Eigen::Matrix4d camera_to_lidar = data_loader.getCameraToLiDARTransform(0);
        
        // Perception: Detect objects
        profiler.startTimer("perception");
        std::vector<Detection> detections;
        
        if (!image.empty() && point_cloud && !point_cloud->empty()) {
            // Multimodal detection
            detections = perception.processFrame(image, point_cloud, camera_matrix, 
                                               camera_to_lidar, timestamp);
        } else if (!image.empty()) {
            // Camera-only detection (simplified)
            std::cout << "\nCamera-only mode not fully implemented in this demo" << std::endl;
        } else if (point_cloud && !point_cloud->empty()) {
            // LiDAR-only detection
            LiDARDetector lidar_detector;
            lidar_detector.initialize();
            detections = lidar_detector.detect(point_cloud);
            for (auto& det : detections) {
                det.timestamp = timestamp;
            }
        }
        profiler.stopTimer("perception");
        
        // Fusion: Update tracks
        profiler.startTimer("fusion");
        fusion.processFrame(detections, imu_data, camera_matrix, timestamp);
        auto active_tracks = fusion.getActiveTracks();
        profiler.stopTimer("fusion");
        
        // Visualization
        profiler.startTimer("visualization");
        
        // 2D Visualization
        if (!image.empty()) {
            cv::Mat annotated_image = visualizer.visualizeImage(image, active_tracks, 
                                                              detections, camera_matrix, true);
            
            // Add statistics overlay
            auto stats = fusion.getTrackingStats();
            annotated_image = visualizer.displayStatistics(annotated_image, stats);
            
            // Display image
            cv::imshow("Multimodal Tracking", annotated_image);
        }
        
        // 3D Visualization
        if (point_cloud && !point_cloud->empty()) {
            Eigen::Vector3d ego_position = imu_data.position;
            visualizer.visualize3D(point_cloud, active_tracks, detections, ego_position);
            visualizer.update3DVisualization(true);
        }
        
        profiler.stopTimer("visualization");
        profiler.stopTimer("total_frame");
        
        // Handle user input
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) { // 'q' or ESC
            break;
        } else if (key == 'p') {
            paused = !paused;
            std::cout << (paused ? "\nPaused" : "\nResumed") << std::endl;
        } else if (key == 's') {
            std::string save_path = "tracking_result_" + std::to_string(frame_index) + ".png";
            if (!image.empty()) {
                cv::Mat save_image = visualizer.visualizeImage(image, active_tracks, 
                                                             detections, camera_matrix, true);
                cv::imwrite(save_path, save_image);
                std::cout << "\nSaved frame to: " << save_path << std::endl;
            }
        }
        
        if (!paused) {
            frame_index++;
        }
        
        // Add small delay for real-time visualization
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    std::cout << "\n\nProcessing completed!" << std::endl;
    
    // Print final statistics
    auto final_stats = fusion.getTrackingStats();
    std::cout << "\n=== Final Tracking Statistics ===" << std::endl;
    std::cout << "Total frames processed: " << frame_index << std::endl;
    std::cout << "Active tracks: " << final_stats.active_tracks << std::endl;
    std::cout << "Total tracks created: " << final_stats.total_tracks << std::endl;
    std::cout << "Average tracking FPS: " << final_stats.tracking_fps << std::endl;
    
    // Print performance summary
    profiler.printSummary();
    
    // Keep 3D viewer open
    if (visualizer.initialize("", true)) {
        std::cout << "\nPress any key in the image window to exit..." << std::endl;
        cv::waitKey(0);
    }
    
    cv::destroyAllWindows();
    return 0;
}

// Alternative simplified main for testing without full KITTI dataset
int main_simple() {
    std::cout << "=== Simplified Tracking Demo ===" << std::endl;
    
    // Create synthetic data for testing
    FusionModule fusion;
    fusion.initialize();
    
    Visualizer visualizer;
    visualizer.initialize("Simple Demo", false);
    
    // Create some synthetic detections
    std::vector<Detection> detections;
    for (int i = 0; i < 3; ++i) {
        Detection det;
        det.center_3d << 10.0 + i * 5.0, 0.0, 0.0;
        det.size_3d << 2.0, 4.0, 1.5;
        det.confidence = 0.8;
        det.class_id = 0; // Car
        det.timestamp = i * 0.1;
        detections.push_back(det);
    }
    
    // Process detections
    IMUData imu_data;
    Eigen::Matrix3d camera_matrix = Eigen::Matrix3d::Identity();
    camera_matrix(0, 0) = 721.5377; // fx
    camera_matrix(1, 1) = 721.5377; // fy
    camera_matrix(0, 2) = 609.5593; // cx
    camera_matrix(1, 2) = 172.8540; // cy
    
    fusion.processFrame(detections, imu_data, camera_matrix, 0.1);
    
    auto tracks = fusion.getActiveTracks();
    std::cout << "Created " << tracks.size() << " tracks" << std::endl;
    
    for (auto* track : tracks) {
        Eigen::VectorXd state = track->ekf->getState();
        std::cout << "Track " << track->id << " position: " 
                  << state.head<3>().transpose() << std::endl;
    }
    
    return 0;
}
