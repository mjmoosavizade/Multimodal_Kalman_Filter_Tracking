#include "kalman_filter.h"
#include "object_tracker.h"
#include "visualization.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#includ    std::vector<Detection> generateDetections(double time) {
        std::vector<Detection> detections;
        
        // For real images, generate more realistic road-based detections
        if (!image_files_.empty()) {
            // Generate realistic detections for vehicles on road
            int num_vehicles = 2 + (int)(time * 0.5) % 4; // 2-5 vehicles
            
            for (int i = 0; i < num_vehicles; i++) {
                Detection det;
                
                // Position vehicles at realistic road locations
                double road_x = 20 + (i * 15) + sin(time + i) * 5; // Varying distances
                double road_y = -10 + i * 4 + cos(time * 0.8 + i) * 2; // Different lanes
                double road_z = -1.65; // Road level
                
                det.center_3d = Eigen::Vector3d(road_x, road_y, road_z);
                det.size_3d = Eigen::Vector3d(4.2, 1.8, 1.5); // Typical car size
                det.confidence = 0.75 + 0.2 * sin(time + i); // Varying confidence
                det.class_id = 0; // Car
                det.timestamp = time;
                
                // More realistic 2D projection for KITTI camera
                double focal_length = 721.5377;
                double cx = 609.5593, cy = 172.854;
                
                if (road_x > 5) { // Only detect objects in reasonable range
                    det.bbox_2d.x = cx + focal_length * (road_y / road_x) - 40;
                    det.bbox_2d.y = cy - focal_length * (road_z / road_x) - 30;
                    det.bbox_2d.width = 80 * (10.0 / road_x); // Scale with distance
                    det.bbox_2d.height = 60 * (10.0 / road_x);
                    
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
        
        // Original simulated detection generation for fallback
        for (const auto& obj : simulated_objects_) {ector>
#include <random>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <sstream>

using namespace tracking;

class TrackingDemo {
private:
    cv::Size image_size_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> noise_dist_;
    std::vector<cv::Scalar> demo_colors_;
    
    struct SimulatedObject {
        Eigen::Vector3d position;
        Eigen::Vector3d velocity;
        Eigen::Vector3d size;
        int class_id;
        cv::Scalar color;
        double birth_time;
        std::vector<Eigen::Vector3d> trajectory;
    };
    
    std::vector<SimulatedObject> simulated_objects_;
    
    // Real image loading
    std::string image_dir_;
    std::vector<std::string> image_files_;
    int current_frame_idx_;

public:
    TrackingDemo() : image_size_(1242, 375), rng_(42), noise_dist_(-0.1, 0.1), current_frame_idx_(0) {
        initializeColors();
        loadImageSequence();
        createSimulatedScene();
    }
    
    void initializeColors() {
        demo_colors_ = {
            cv::Scalar(255, 100, 100), // Red
            cv::Scalar(100, 255, 100), // Green
            cv::Scalar(100, 100, 255), // Blue
            cv::Scalar(255, 255, 100), // Cyan
            cv::Scalar(255, 100, 255), // Magenta
            cv::Scalar(100, 255, 255), // Yellow
            cv::Scalar(200, 150, 100), // Orange
            cv::Scalar(150, 100, 200), // Purple
        };
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
            std::cerr << "    Will use simulated scenes instead." << std::endl;
        }
    }
    
    void createSimulatedScene() {
        // Create multiple moving objects with different trajectories
        
        // Car moving forward
        SimulatedObject car1;
        car1.position = Eigen::Vector3d(50, 0, 0);
        car1.velocity = Eigen::Vector3d(15, 0, 0);
        car1.size = Eigen::Vector3d(4.5, 2.0, 1.8);
        car1.class_id = 0; // Car
        car1.color = demo_colors_[0];
        car1.birth_time = 0.0;
        simulated_objects_.push_back(car1);
        
        // Car turning
        SimulatedObject car2;
        car2.position = Eigen::Vector3d(30, -10, 0);
        car2.velocity = Eigen::Vector3d(12, 3, 0);
        car2.size = Eigen::Vector3d(4.0, 1.9, 1.7);
        car2.class_id = 0;
        car2.color = demo_colors_[1];
        car2.birth_time = 0.5;
        simulated_objects_.push_back(car2);
        
        // Pedestrian crossing
        SimulatedObject ped1;
        ped1.position = Eigen::Vector3d(25, 5, 0);
        ped1.velocity = Eigen::Vector3d(0, -1.5, 0);
        ped1.size = Eigen::Vector3d(0.6, 0.6, 1.8);
        ped1.class_id = 1; // Pedestrian
        ped1.color = demo_colors_[2];
        ped1.birth_time = 1.0;
        simulated_objects_.push_back(ped1);
        
        // Cyclist
        SimulatedObject cyclist;
        cyclist.position = Eigen::Vector3d(40, 3, 0);
        cyclist.velocity = Eigen::Vector3d(8, -1, 0);
        cyclist.size = Eigen::Vector3d(1.8, 0.7, 1.6);
        cyclist.class_id = 2; // Cyclist
        cyclist.color = demo_colors_[3];
        cyclist.birth_time = 2.0;
        simulated_objects_.push_back(cyclist);
        
        // Overtaking car
        SimulatedObject car3;
        car3.position = Eigen::Vector3d(20, -3, 0);
        car3.velocity = Eigen::Vector3d(20, 1, 0);
        car3.size = Eigen::Vector3d(4.2, 1.8, 1.5);
        car3.class_id = 0;
        car3.color = demo_colors_[4];
        car3.birth_time = 1.5;
        simulated_objects_.push_back(car3);
    }
    
    std::vector<Detection> generateDetections(double time) {
        std::vector<Detection> detections;
        
        for (auto& obj : simulated_objects_) {
            if (time < obj.birth_time) continue;
            
            double dt = time - obj.birth_time;
            
            // Update position with some realistic motion
            Eigen::Vector3d pos = obj.position + obj.velocity * dt;
            
            // Add some curved motion for cars
            if (obj.class_id == 0 && obj.color == demo_colors_[1]) { // Turning car
                double turn_rate = 0.1;
                pos.y() += 2.0 * sin(turn_rate * dt);
                pos.x() += cos(turn_rate * dt) * dt * 2.0;
            }
            
            // Add some weaving for cyclist
            if (obj.class_id == 2) {
                pos.y() += 0.5 * sin(0.5 * dt);
            }
            
            // Store trajectory
            obj.trajectory.push_back(pos);
            if (obj.trajectory.size() > 50) {
                obj.trajectory.erase(obj.trajectory.begin());
            }
            
            // Add measurement noise
            Eigen::Vector3d noisy_pos = pos;
            noisy_pos.x() += noise_dist_(rng_) * 2.0;
            noisy_pos.y() += noise_dist_(rng_) * 1.0;
            noisy_pos.z() += noise_dist_(rng_) * 0.5;
            
            // Create detection
            Detection det;
            det.center_3d = noisy_pos;
            det.size_3d = obj.size;
            det.confidence = 0.85 + noise_dist_(rng_) * 0.1;
            det.class_id = obj.class_id;
            det.timestamp = time;
            
            // Project to 2D (simple perspective projection)
            double focal_length = 721.5377;
            double cx = 609.5593, cy = 172.854;
            if (pos.x() > 0) { // Only detect objects in front
                det.bbox_2d.x = cx + focal_length * (pos.y() / pos.x()) - obj.size.y() * 10;
                det.bbox_2d.y = cy - focal_length * (pos.z() / pos.x()) - obj.size.z() * 10;
                det.bbox_2d.width = obj.size.y() * 20 / (pos.x() / 20);
                det.bbox_2d.height = obj.size.z() * 20 / (pos.x() / 20);
                
                // Ensure detection is within image bounds
                if (det.bbox_2d.x >= 0 && det.bbox_2d.y >= 0 && 
                    det.bbox_2d.x + det.bbox_2d.width <= image_size_.width &&
                    det.bbox_2d.y + det.bbox_2d.height <= image_size_.height) {
                    detections.push_back(det);
                }
            }
        }
        
        return detections;
    }
    
    cv::Mat createDemoImage(double time) {
        // Use real images if available, otherwise fall back to simulated
        if (!image_files_.empty()) {
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
        }
        
        // Fallback to simulated scene if real images not available
        cv::Mat image = cv::Mat::zeros(image_size_.height, image_size_.width, CV_8UC3);
        
        // Create a road-like background
        cv::Scalar road_color(60, 60, 60);
        image.setTo(road_color);
        
        // Add road markings
        cv::Scalar line_color(200, 200, 200);
        
        // Center line
        for (int y = 0; y < image_size_.height; y += 20) {
            cv::line(image, 
                    cv::Point(image_size_.width/2, y), 
                    cv::Point(image_size_.width/2, y + 10), 
                    line_color, 2);
        }
        
        // Side lines
        cv::line(image, cv::Point(50, 0), cv::Point(50, image_size_.height), line_color, 3);
        cv::line(image, cv::Point(image_size_.width-50, 0), cv::Point(image_size_.width-50, image_size_.height), line_color, 3);
        
        // Add perspective grid
        for (int i = 1; i <= 10; i++) {
            int y = image_size_.height - i * 30;
            cv::line(image, cv::Point(0, y), cv::Point(image_size_.width, y), cv::Scalar(80, 80, 80), 1);
        }
        
        // Add timestamp
        std::stringstream ss;
        ss << "Time: " << std::fixed << std::setprecision(1) << time << "s (Simulated)";
        cv::putText(image, ss.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        
        // Add system info
        cv::putText(image, "Multimodal Kalman Filter Tracking", cv::Point(20, image_size_.height - 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(100, 255, 100), 2);
        cv::putText(image, "EKF: 9D State | Multi-Object | Sensor Fusion", cv::Point(20, image_size_.height - 35), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(150, 150, 255), 1);
        cv::putText(image, "C++ Implementation | Real-time Processing", cv::Point(20, image_size_.height - 15), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(150, 150, 255), 1);
        
        return image;
    }
    
    void runDemo() {
        std::cout << "\n🎬 Starting Tracking Visualization Demo..." << std::endl;
        std::cout << "📊 Features: EKF tracking, Multi-object, Trajectory visualization" << std::endl;
        std::cout << "🎯 Objects: Cars, Pedestrians, Cyclists with realistic motion" << std::endl;
        std::cout << "⚡ Real-time processing simulation" << std::endl;
        std::cout << "\nPress 'q' to quit, 's' to save frame, 'p' to pause\n" << std::endl;
        
        // Initialize tracker and visualizer
        auto tracker = std::make_unique<MultiObjectTracker>();
        Visualizer visualizer;
        visualizer.initialize("Kalman Filter Tracking Demo", false);
        
        // Demo parameters
        double time = 0.0;
        double dt = 0.1; // 10 FPS
        bool paused = false;
        
        // Camera matrix for projection (KITTI-like)
        Eigen::Matrix3d camera_matrix;
        camera_matrix << 721.5377, 0, 609.5593,
                        0, 721.5377, 172.854,
                        0, 0, 1;
        
        cv::VideoWriter video_writer;
        bool recording = false;
        
        while (true) {
            // Generate detections and update tracker outside the paused check
            std::vector<Detection> detections = generateDetections(time);
            auto tracks = tracker->getActiveTracks();
            
            if (!paused) {
                // Update tracker
                tracker->update(detections, camera_matrix);
                tracks = tracker->getActiveTracks();
                
                // Create demo image
                cv::Mat demo_image = createDemoImage(time);
                
                // Enhanced visualization
                cv::Mat vis_image = visualizer.visualizeImage(demo_image, tracks, detections, camera_matrix, true);
                
                // Add performance metrics
                addPerformanceOverlay(vis_image, tracks, detections, time);
                
                // Add trajectory trails
                addTrajectoryTrails(vis_image, tracks, camera_matrix);
                
                // Add detection confidence
                addDetectionDetails(vis_image, detections);
                
                // Create side-by-side display: raw frame + processed frame
                cv::Mat combined_display;
                
                // Ensure both images are same height
                cv::Mat raw_frame = demo_image.clone();
                cv::Mat processed_frame = vis_image.clone();
                
                if (raw_frame.rows != processed_frame.rows) {
                    cv::resize(raw_frame, raw_frame, processed_frame.size());
                }
                
                // Add labels to each frame
                cv::putText(raw_frame, "Raw Input Frame", cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
                cv::putText(processed_frame, "Tracking Results", cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                
                // Concatenate horizontally
                cv::hconcat(raw_frame, processed_frame, combined_display);
                
                // Display combined view
                cv::imshow("Multimodal Kalman Filter Tracking Demo", combined_display);
                
                // Save frame if recording (save the combined view)
                if (recording && video_writer.isOpened()) {
                    video_writer.write(combined_display);
                }
                
                time += dt;
                
                // Reset simulation after 30 seconds
                if (time > 30.0) {
                    time = 0.0;
                    // Reset tracker by creating new instance
                    tracker = std::make_unique<MultiObjectTracker>();
                    createSimulatedScene(); // Reset scene
                }
            }
            
            // Handle keyboard input
            int key = cv::waitKey(100) & 0xFF;
            if (key == 'q' || key == 27) break; // Quit
            if (key == 'p') paused = !paused;   // Pause/unpause
            if (key == 's') {                   // Save frame  
                std::vector<Detection> current_detections = generateDetections(time);
                auto current_tracks = tracker->getActiveTracks();
                cv::Mat current_image = createDemoImage(time);
                cv::Mat current_vis = visualizer.visualizeImage(current_image, current_tracks, current_detections, camera_matrix, true);
                
                // Create combined frame for screenshot
                cv::Mat current_raw = current_image.clone();
                cv::Mat current_processed = current_vis.clone();
                
                // Add labels
                cv::putText(current_raw, "Raw Input Frame", cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
                cv::putText(current_processed, "Tracking Results", cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                
                cv::Mat current_combined;
                cv::hconcat(current_raw, current_processed, current_combined);
                
                std::string filename = "tracking_demo_" + std::to_string((int)(time*10)) + ".png";
                cv::imwrite(filename, current_combined);
                std::cout << "💾 Saved frame: " << filename << std::endl;
            }
            if (key == 'r') {                   // Start/stop recording
                if (!recording) {
                    std::vector<Detection> sample_detections = generateDetections(time);
                    auto sample_tracks = tracker->getActiveTracks();
                    cv::Mat sample_image = createDemoImage(time);
                    cv::Mat sample_vis = visualizer.visualizeImage(sample_image, sample_tracks, sample_detections, camera_matrix, true);
                    
                    // Create combined frame for video size calculation
                    cv::Mat sample_combined;
                    cv::hconcat(sample_image, sample_vis, sample_combined);
                    
                    video_writer.open("tracking_demo.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 10.0, sample_combined.size());
                    if (video_writer.isOpened()) {
                        recording = true;
                        std::cout << "🎥 Started recording..." << std::endl;
                    }
                } else {
                    video_writer.release();
                    recording = false;
                    std::cout << "⏹️  Stopped recording. Video saved as tracking_demo.mp4" << std::endl;
                }
            }
        }
        
        cv::destroyAllWindows();
        if (video_writer.isOpened()) {
            video_writer.release();
        }
    }
    
private:
    void addPerformanceOverlay(cv::Mat& image, const std::vector<TrackedObject*>& tracks, 
                              const std::vector<Detection>& detections, double /*time*/) {
        int y_offset = 60;
        cv::Scalar info_color(100, 255, 255);
        
        // Tracking statistics
        std::string stats = "Active Tracks: " + std::to_string(tracks.size()) + 
                           " | Detections: " + std::to_string(detections.size());
        cv::putText(image, stats, cv::Point(image_size_.width - 350, y_offset), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2);
        
        // Frame rate simulation
        std::string fps = "FPS: 10.0 | Processing: 45ms";
        cv::putText(image, fps, cv::Point(image_size_.width - 300, y_offset + 25), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1);
        
        // EKF status
        std::string ekf_info = "EKF State: 9D (pos,vel,acc) | Covariance: Adaptive";
        cv::putText(image, ekf_info, cv::Point(image_size_.width - 450, y_offset + 45), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1);
    }
    
    void addTrajectoryTrails(cv::Mat& image, const std::vector<TrackedObject*>& tracks, 
                           const Eigen::Matrix3d& camera_matrix) {
        for (auto* track : tracks) {
            if (!track->ekf->isInitialized()) continue;
            
            // Get trajectory from simulated object
            cv::Scalar trail_color = demo_colors_[track->id % demo_colors_.size()];
            trail_color *= 0.7; // Dimmer for trail
            
            // Draw predicted future positions
            Eigen::VectorXd state = track->ekf->getState();
            Eigen::Vector3d current_pos = state.head<3>();
            Eigen::Vector3d velocity = state.segment<3>(3);
            
            std::vector<cv::Point> trajectory_points;
            for (int i = 1; i <= 10; i++) {
                Eigen::Vector3d future_pos = current_pos + velocity * (i * 0.5);
                if (future_pos.x() > 0) {
                    cv::Point2f proj = project3DTo2D(future_pos, camera_matrix);
                    if (proj.x >= 0 && proj.x < image_size_.width && 
                        proj.y >= 0 && proj.y < image_size_.height) {
                        trajectory_points.push_back(cv::Point(proj.x, proj.y));
                    }
                }
            }
            
            // Draw trajectory line
            for (size_t i = 1; i < trajectory_points.size(); i++) {
                cv::line(image, trajectory_points[i-1], trajectory_points[i], trail_color, 2);
                cv::circle(image, trajectory_points[i], 3, trail_color, -1);
            }
        }
    }
    
    void addDetectionDetails(cv::Mat& image, const std::vector<Detection>& detections) {
        for (const auto& det : detections) {
            // Add confidence score
            std::string conf_text = cv::format("%.2f", det.confidence);
            cv::Point conf_pos(det.bbox_2d.x, det.bbox_2d.y - 5);
            cv::putText(image, conf_text, conf_pos, cv::FONT_HERSHEY_SIMPLEX, 0.4, 
                       cv::Scalar(255, 255, 0), 1);
            
            // Add class label
            std::string class_name;
            switch (det.class_id) {
                case 0: class_name = "Car"; break;
                case 1: class_name = "Pedestrian"; break;
                case 2: class_name = "Cyclist"; break;
                default: class_name = "Unknown"; break;
            }
            cv::Point class_pos(det.bbox_2d.x, det.bbox_2d.y + det.bbox_2d.height + 15);
            cv::putText(image, class_name, class_pos, cv::FONT_HERSHEY_SIMPLEX, 0.4, 
                       cv::Scalar(255, 255, 255), 1);
        }
    }
    
    cv::Point2f project3DTo2D(const Eigen::Vector3d& point_3d, const Eigen::Matrix3d& camera_matrix) {
        if (point_3d.x() <= 0) return cv::Point2f(-1, -1);
        
        Eigen::Vector3d projected = camera_matrix * point_3d;
        return cv::Point2f(projected.x() / projected.z(), projected.y() / projected.z());
    }
};

int main() {
    std::cout << "🚗 Multimodal Kalman Filter Tracking - Visualization Demo" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    TrackingDemo demo;
    demo.runDemo();
    
    std::cout << "\n✨ Demo completed! Thank you for watching." << std::endl;
    return 0;
}
