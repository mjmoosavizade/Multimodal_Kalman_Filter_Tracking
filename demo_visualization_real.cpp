#include "kalman_filter.h"
#include "object_tracker.h"
#include "visualization.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <sstream>

using namespace tracking;

class RealImageDemo {
private:
    cv::Size image_size_;
    std::string image_dir_;
    std::vector<std::string> image_files_;
    int current_frame_idx_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> noise_dist_;
    
public:
    RealImageDemo() : image_size_(1242, 375), current_frame_idx_(0), rng_(42), noise_dist_(-0.1, 0.1) {
        loadImageSequence();
    }
    
    void loadImageSequence() {
        image_dir_ = "data/image_00/data/";
        
        // Load all available image files
        for (int i = 0; i < 200; i++) { // Check up to 200 frames
            std::stringstream ss;
            ss << image_dir_ << std::setfill('0') << std::setw(10) << i << ".png";
            std::string filepath = ss.str();
            
            // Check if file exists
            std::ifstream file(filepath);
            if (file.good()) {
                image_files_.push_back(filepath);
                file.close();
            } else {
                break; // Stop when no more consecutive files found
            }
        }
        
        std::cout << "📁 Loaded " << image_files_.size() << " image frames from " << image_dir_ << std::endl;
        
        if (image_files_.empty()) {
            std::cerr << "❌ Warning: No images found in " << image_dir_ << std::endl;
        }
    }
    
    cv::Mat loadRealImage(double time) {
        if (image_files_.empty()) {
            return cv::Mat::zeros(image_size_.height, image_size_.width, CV_8UC3);
        }
        
        // Calculate frame index based on time (10 FPS)
        int frame_idx = static_cast<int>(time * 10) % image_files_.size();
        current_frame_idx_ = frame_idx;
        
        cv::Mat image = cv::imread(image_files_[frame_idx]);
        if (!image.empty()) {
            // Add timestamp overlay
            std::stringstream ss;
            ss << "Frame: " << frame_idx << " | Time: " << std::fixed << std::setprecision(1) << time << "s";
            cv::putText(image, ss.str(), cv::Point(20, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            
            // Add frame info
            ss.str("");
            ss << "Real KITTI Data | " << image_files_.size() << " frames loaded";
            cv::putText(image, ss.str(), cv::Point(20, image.rows - 20), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
            
            return image;
        }
        
        return cv::Mat::zeros(image_size_.height, image_size_.width, CV_8UC3);
    }
    
    std::vector<Detection> generateRealisticDetections(double time) {
        std::vector<Detection> detections;
        
        // Generate realistic detections for vehicles on road
        int num_vehicles = 2 + (int)(time * 0.3) % 4; // 2-5 vehicles
        
        for (int i = 0; i < num_vehicles; i++) {
            Detection det;
            
            // Position vehicles at realistic road locations
            double road_x = 15 + (i * 12) + sin(time * 0.5 + i) * 3; // Varying distances
            double road_y = -8 + i * 3.5 + cos(time * 0.4 + i) * 1.5; // Different lanes
            double road_z = -1.65; // Road level
            
            det.center_3d = Eigen::Vector3d(road_x, road_y, road_z);
            det.size_3d = Eigen::Vector3d(4.2, 1.8, 1.5); // Typical car size
            det.confidence = 0.75 + 0.15 * sin(time + i); // Varying confidence
            det.class_id = 0; // Car
            det.timestamp = time;
            
            // KITTI camera calibration parameters
            double focal_length = 721.5377;
            double cx = 609.5593, cy = 172.854;
            
            if (road_x > 5) { // Only detect objects in reasonable range
                det.bbox_2d.x = cx + focal_length * (road_y / road_x) - 35;
                det.bbox_2d.y = cy - focal_length * (road_z / road_x) - 25;
                det.bbox_2d.width = 70 * (12.0 / road_x); // Scale with distance
                det.bbox_2d.height = 50 * (12.0 / road_x);
                
                // Add some noise
                det.bbox_2d.x += noise_dist_(rng_) * 5;
                det.bbox_2d.y += noise_dist_(rng_) * 3;
                
                // Ensure bbox is within image bounds
                if (det.bbox_2d.x > 0 && det.bbox_2d.y > 0 && 
                    det.bbox_2d.x + det.bbox_2d.width < 1242 && 
                    det.bbox_2d.y + det.bbox_2d.height < 375) {
                    detections.push_back(det);
                }
            }
        }
        
        return detections;
    }
    
    void runDemo() {
        if (image_files_.empty()) {
            std::cerr << "❌ Cannot run demo: No images found!" << std::endl;
            return;
        }
        
        std::cout << "🎬 Starting Real Image Tracking Demo..." << std::endl;
        std::cout << "📊 Features: Real KITTI images, EKF tracking, Professional visualization" << std::endl;
        std::cout << "🚗 Processing " << image_files_.size() << " real camera frames" << std::endl;
        std::cout << "⚡ Press 'q' to quit, 's' to save frame, 'p' to pause" << std::endl;
        
        // Initialize tracking components
        auto tracker = std::make_unique<MultiObjectTracker>();
        tracking::Visualizer visualizer;
        
        // Camera matrix for KITTI dataset
        Eigen::Matrix3d camera_matrix;
        camera_matrix << 721.5377, 0, 609.5593,
                         0, 721.5377, 172.854,
                         0, 0, 1;
        
        // Demo control variables
        double time = 0.0;
        bool paused = false;
        bool recording = false;
        cv::VideoWriter video_writer;
        
        std::cout << "Visualizer initialized with window: Real KITTI Tracking Demo" << std::endl;
        
        while (true) {
            if (!paused) {
                time += 0.1; // 10 FPS
                
                // Load real image
                cv::Mat real_image = loadRealImage(time);
                
                // Generate realistic detections for the scene
                std::vector<Detection> detections = generateRealisticDetections(time);
                
                // Update tracker
                tracker->update(detections, camera_matrix);
                auto tracks = tracker->getActiveTracks();
                
                // Create visualization
                cv::Mat vis_image = visualizer.visualizeImage(real_image, tracks, detections, camera_matrix, true);
                
                // Create side-by-side display: raw frame + processed frame
                cv::Mat raw_frame = real_image.clone();
                cv::Mat processed_frame = vis_image.clone();
                
                // Add labels to each frame
                cv::putText(raw_frame, "Raw KITTI Frame", cv::Point(10, 50), 
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
                cv::putText(processed_frame, "Tracking Results", cv::Point(10, 50), 
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                
                // Concatenate horizontally
                cv::Mat combined_display;
                cv::hconcat(raw_frame, processed_frame, combined_display);
                
                // Display
                cv::imshow("Real KITTI Tracking Demo", combined_display);
                
                // Save frame if recording
                if (recording && video_writer.isOpened()) {
                    video_writer.write(combined_display);
                }
                
                // Auto-reset after all frames shown
                if (current_frame_idx_ >= image_files_.size() - 1) {
                    time = 0.0;
                    tracker = std::make_unique<MultiObjectTracker>();
                }
            }
            
            // Handle keyboard input
            int key = cv::waitKey(100) & 0xFF;
            if (key == 'q' || key == 27) break; // Quit
            if (key == 'p') paused = !paused;   // Pause/unpause
            if (key == 's') {                   // Save frame  
                cv::Mat current_image = loadRealImage(time);
                std::vector<Detection> current_detections = generateRealisticDetections(time);
                auto current_tracks = tracker->getActiveTracks();
                cv::Mat current_vis = visualizer.visualizeImage(current_image, current_tracks, current_detections, camera_matrix, true);
                
                cv::Mat current_combined;
                cv::hconcat(current_image, current_vis, current_combined);
                
                std::string filename = "kitti_tracking_" + std::to_string(current_frame_idx_) + ".png";
                cv::imwrite(filename, current_combined);
                std::cout << "💾 Saved frame: " << filename << std::endl;
            }
            if (key == 'r') {                   // Start/stop recording
                if (!recording) {
                    cv::Mat sample_image = loadRealImage(time);
                    cv::Mat sample_combined;
                    cv::hconcat(sample_image, sample_image, sample_combined); // Temporary size
                    
                    video_writer.open("kitti_tracking_demo.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 10.0, sample_combined.size());
                    if (video_writer.isOpened()) {
                        recording = true;
                        std::cout << "🎥 Started recording..." << std::endl;
                    }
                } else {
                    video_writer.release();
                    recording = false;
                    std::cout << "⏹️  Stopped recording. Video saved as kitti_tracking_demo.mp4" << std::endl;
                }
            }
        }
        
        cv::destroyAllWindows();
        if (video_writer.isOpened()) {
            video_writer.release();
        }
        
        std::cout << "🏁 Demo completed!" << std::endl;
    }
};

int main() {
    try {
        RealImageDemo demo;
        demo.runDemo();
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
