#include "kalman_filter.h"
#include "object_tracker.h"
#include "fusion_module.h"
#include <iostream>
#include <vector>

using namespace tracking;

/**
 * Simple demonstration of the tracking system without requiring KITTI dataset
 */
int main() {
    std::cout << "=== Simple Tracking Demo ===" << std::endl;
    
    // Initialize fusion module
    FusionModule fusion;
    fusion.initialize(10, 3.0, 3); // max_tracks, threshold, max_misses
    
    // Create synthetic camera matrix
    Eigen::Matrix3d camera_matrix = Eigen::Matrix3d::Identity();
    camera_matrix(0, 0) = 721.5377; // fx
    camera_matrix(1, 1) = 721.5377; // fy
    camera_matrix(0, 2) = 609.5593; // cx
    camera_matrix(1, 2) = 172.8540; // cy
    
    // Simulate tracking over multiple frames
    for (int frame = 0; frame < 10; ++frame) {
        std::cout << "\n--- Frame " << frame << " ---" << std::endl;
        
        // Create synthetic detections (simulating moving objects)
        std::vector<Detection> detections;
        
        // Object 1: Moving forward
        Detection det1;
        det1.center_3d << 10.0 + frame * 2.0, 0.0, 0.0;
        det1.size_3d << 2.0, 4.0, 1.5;
        det1.confidence = 0.9;
        det1.class_id = 0; // Car
        det1.timestamp = frame * 0.1;
        det1.bbox_2d = cv::Rect2d(100 + frame * 10, 200, 80, 60);
        detections.push_back(det1);
        
        // Object 2: Moving diagonally (appears after frame 3)
        if (frame >= 3) {
            Detection det2;
            det2.center_3d << 5.0 + (frame - 3) * 1.5, 2.0 + (frame - 3) * 1.0, 0.0;
            det2.size_3d << 1.8, 3.5, 1.4;
            det2.confidence = 0.8;
            det2.class_id = 0; // Car
            det2.timestamp = frame * 0.1;
            det2.bbox_2d = cv::Rect2d(150 + (frame - 3) * 8, 180 + (frame - 3) * 5, 75, 55);
            detections.push_back(det2);
        }
        
        // Object 3: Stationary object (disappears after frame 6)
        if (frame <= 6) {
            Detection det3;
            det3.center_3d << 20.0, -5.0, 0.0;
            det3.size_3d << 0.6, 0.6, 1.7;
            det3.confidence = 0.7;
            det3.class_id = 1; // Pedestrian
            det3.timestamp = frame * 0.1;
            det3.bbox_2d = cv::Rect2d(300, 250, 30, 80);
            detections.push_back(det3);
        }
        
        // Create synthetic IMU data
        IMUData imu_data;
        imu_data.timestamp = frame * 0.1;
        imu_data.velocity << 0.0, 0.0, 0.0; // Ego vehicle stationary
        imu_data.linear_acceleration << 0.0, 0.0, 0.0;
        imu_data.position << 0.0, 0.0, 0.0;
        
        // Process frame
        fusion.processFrame(detections, imu_data, camera_matrix, frame * 0.1);
        
        // Get tracking results
        auto active_tracks = fusion.getActiveTracks();
        
        std::cout << "Detections: " << detections.size() << std::endl;
        std::cout << "Active tracks: " << active_tracks.size() << std::endl;
        
        // Display track information
        for (auto* track : active_tracks) {
            Eigen::VectorXd state = track->ekf->getState();
            std::cout << "  Track " << track->id 
                      << " (class " << track->class_id << ")"
                      << " pos: [" << state(0) << ", " << state(1) << ", " << state(2) << "]"
                      << " vel: [" << state(3) << ", " << state(4) << ", " << state(5) << "]"
                      << " conf: " << track->confidence
                      << std::endl;
        }
        
        // Get predictions for next frame
        auto predictions = fusion.predictTracks(0.1);
        if (!predictions.empty()) {
            std::cout << "  Predictions for next frame:" << std::endl;
            for (size_t i = 0; i < predictions.size(); ++i) {
                std::cout << "    Track " << i 
                          << " predicted pos: [" 
                          << predictions[i](0) << ", " 
                          << predictions[i](1) << ", " 
                          << predictions[i](2) << "]" << std::endl;
            }
        }
    }
    
    // Display final statistics
    auto stats = fusion.getTrackingStats();
    std::cout << "\n=== Final Statistics ===" << std::endl;
    std::cout << "Total tracks created: " << stats.total_tracks << std::endl;
    std::cout << "Active tracks: " << stats.active_tracks << std::endl;
    std::cout << "Average track age: " << stats.average_track_age << " seconds" << std::endl;
    std::cout << "Tracking FPS: " << stats.tracking_fps << std::endl;
    
    return 0;
}
