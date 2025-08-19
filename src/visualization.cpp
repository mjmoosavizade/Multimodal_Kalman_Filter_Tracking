#include "visualization.h"
#ifdef WITH_PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#endif
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>

namespace tracking {

Visualizer::Visualizer() : enable_3d_viewer_(false) {
    initializeColorPalette();
}

bool Visualizer::initialize(const std::string& window_name, bool enable_3d_viewer) {
    window_name_ = window_name;
    enable_3d_viewer_ = enable_3d_viewer;
    
    if (enable_3d_viewer_) {
    #ifdef WITH_PCL
    viewer_3d_ = std::make_shared<pcl::visualization::PCLVisualizer>("3D Tracking Viewer");
    viewer_3d_->setBackgroundColor(0, 0, 0);
    viewer_3d_->addCoordinateSystem(1.0);
    viewer_3d_->initCameraParameters();
        
    // Set camera position for better view
    viewer_3d_->setCameraPosition(0, 0, 50, 0, 0, 0, 0, 1, 0);
    #else
    std::cerr << "3D viewer requested but PCL is not available; skipping 3D viewer initialization." << std::endl;
    #endif
    }
    
    std::cout << "Visualizer initialized with window: " << window_name_ << std::endl;
    return true;
}

cv::Mat Visualizer::visualizeImage(const cv::Mat& image,
                                  const std::vector<tracking::TrackedObject*>& tracks,
                                  const std::vector<Detection>& detections,
                                  const Eigen::Matrix3d& camera_matrix,
                                  bool show_predictions) {
    
    cv::Mat annotated_image = image.clone();
    
    if (annotated_image.empty()) {
        return annotated_image;
    }
    
    // Draw current detections in light colors
    for (const auto& detection : detections) {
        if (detection.bbox_2d.area() > 0) {
            cv::Scalar color(100, 100, 100); // Gray for detections
            drawBoundingBox2D(annotated_image, detection.bbox_2d, color, 1, 
                             "Det: " + getClassName(detection.class_id));
        }
    }
    
    // Draw tracked objects
    for (auto* track : tracks) {
        if (!track->ekf->isInitialized()) continue;
        
        cv::Scalar color = getTrackColor(track->id);
        Eigen::VectorXd state = track->ekf->getState();
        
        // Draw 3D bounding box if we have 3D information
        if (state.head<3>().norm() > 0) {
            Eigen::Vector3d center_3d = state.head<3>();
            Eigen::Vector3d size_3d(2.0, 4.0, 1.5); // Default car size
            
            drawBoundingBox3D(annotated_image, center_3d, size_3d, 
                             camera_matrix, color, params_.bbox_thickness);
        }
        
        // Draw velocity vector if enabled
        if (params_.show_velocity_vectors && state.size() >= 6) {
            Eigen::Vector3d position = state.head<3>();
            Eigen::Vector3d velocity = state.segment<3>(3);
            drawVelocityVector(annotated_image, position, velocity, 
                             camera_matrix, color, 2.0);
        }
        
        // Draw track ID and information
        cv::Point2f center_2d = project3DTo2D(state.head<3>(), camera_matrix);
        if (center_2d.x > 0 && center_2d.y > 0 && 
            center_2d.x < annotated_image.cols && center_2d.y < annotated_image.rows) {
            
            std::stringstream ss;
            ss << "ID:" << track->id << " (" << getClassName(track->class_id) << ")";
            ss << " C:" << std::fixed << std::setprecision(2) << track->confidence;
            
            cv::putText(annotated_image, ss.str(), 
                       cv::Point(center_2d.x - 50, center_2d.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, params_.text_scale, color, 
                       params_.text_thickness);
        }
        
        // Draw prediction if enabled
        if (show_predictions) {
            ExtendedKalmanFilter temp_filter = *(track->ekf);
            temp_filter.predict(params_.prediction_time);
            Eigen::VectorXd predicted_state = temp_filter.getState();
            
            cv::Point2f pred_center = project3DTo2D(predicted_state.head<3>(), camera_matrix);
            if (pred_center.x > 0 && pred_center.y > 0 && 
                pred_center.x < annotated_image.cols && pred_center.y < annotated_image.rows) {
                
                cv::circle(annotated_image, pred_center, 5, color, -1);
                cv::putText(annotated_image, "PRED", 
                           cv::Point(pred_center.x + 10, pred_center.y),
                           cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
            }
        }
    }
    
    return annotated_image;
}

void Visualizer::visualize3D(const PointCloudPtr& cloud,
                             const std::vector<tracking::TrackedObject*>& tracks,
                             const std::vector<Detection>& detections,
                             const Eigen::Vector3d& ego_position) {
#ifndef WITH_PCL
    (void)cloud; (void)tracks; (void)detections; (void)ego_position;
    std::cerr << "visualize3D called but PCL is not available; skipping 3D visualization." << std::endl;
    return;
#else
    if (!enable_3d_viewer_ || !viewer_3d_) {
        return;
    }

    // Clear previous visualization
    viewer_3d_->removeAllPointClouds();
    viewer_3d_->removeAllShapes();

    // Add point cloud
    if (cloud && !cloud->empty()) {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> 
            intensity_distribution(cloud, "intensity");
        viewer_3d_->addPointCloud<pcl::PointXYZI>(cloud, intensity_distribution, "cloud");
        viewer_3d_->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
    }
    
    // Add ego vehicle position
    viewer_3d_->addSphere(pcl::PointXYZ(ego_position.x(), ego_position.y(), ego_position.z()),
                         0.5, 1.0, 1.0, 0.0, "ego_vehicle");
    
    // Add detection bounding boxes
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& detection = detections[i];
        if (detection.center_3d.norm() > 0) {
            std::array<double, 3> color = {0.5, 0.5, 0.5}; // Gray for detections
            add3DBoundingBox(detection.center_3d, detection.size_3d, 
                           "detection_" + std::to_string(i), color);
        }
    }
    
    // Add tracked object bounding boxes
    for (auto* track : tracks) {
        if (!track->ekf->isInitialized()) continue;
        
        Eigen::VectorXd state = track->ekf->getState();
        Eigen::Vector3d center_3d = state.head<3>();
        Eigen::Vector3d size_3d(2.0, 4.0, 1.5); // Default size
        
        cv::Scalar bgr_color = getTrackColor(track->id);
        std::array<double, 3> rgb_color = bgrToRgb(bgr_color);
        
        add3DBoundingBox(center_3d, size_3d, 
                        "track_" + std::to_string(track->id), rgb_color);
        
        // Add track ID text
        viewer_3d_->addText3D("ID:" + std::to_string(track->id),
                     pcl::PointXYZ(center_3d.x(), center_3d.y(), center_3d.z() + 2.0),
                     0.5, rgb_color[0], rgb_color[1], rgb_color[2],
                     "text_" + std::to_string(track->id));
    }
#endif
}void Visualizer::update3DVisualization(bool spin_once) {
#ifndef WITH_PCL
    (void)spin_once;
    return;
#else
    if (!enable_3d_viewer_ || !viewer_3d_) {
        return;
    }

    if (spin_once) {
        viewer_3d_->spinOnce(100);
    } else {
        viewer_3d_->spin();
    }
#endif
}

bool Visualizer::saveResults(const std::string& image_path, const std::string& cloud_path) {
    // This would save the current visualization results
    // Implementation depends on specific requirements
    std::cout << "Saving results to: " << image_path << std::endl;
    return true;
}

bool Visualizer::createSummaryVideo(const std::string& output_path, double fps) {
    // Initialize video writer if not already done
    if (!video_writer_.isOpened()) {
        cv::Size frame_size(1920, 1080); // Default size
        int fourcc = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
        video_writer_.open(output_path, fourcc, fps, frame_size);
        
        if (!video_writer_.isOpened()) {
            std::cerr << "Failed to open video writer: " << output_path << std::endl;
            return false;
        }
    }
    
    return true;
}

cv::Mat Visualizer::displayStatistics(const cv::Mat& image,
                                     const FusionModule::TrackingStats& stats) {
    
    cv::Mat stats_image = image.clone();
    
    // Create statistics overlay
    std::vector<std::string> stat_lines = {
        "Active Tracks: " + std::to_string(stats.active_tracks),
        "Total Tracks: " + std::to_string(stats.total_tracks),
        "New Tracks: " + std::to_string(stats.new_tracks_this_frame),
        "Lost Tracks: " + std::to_string(stats.lost_tracks_this_frame),
        "Avg Track Age: " + std::to_string(stats.average_track_age) + "s",
        "FPS: " + std::to_string(stats.tracking_fps)
    };
    
    // Draw background rectangle
    cv::Rect stats_rect(10, 10, 300, static_cast<int>(stat_lines.size() * 25 + 20));
    cv::rectangle(stats_image, stats_rect, cv::Scalar(0, 0, 0, 128), -1);
    
    // Draw statistics text
    for (size_t i = 0; i < stat_lines.size(); ++i) {
        cv::putText(stats_image, stat_lines[i],
                   cv::Point(20, 35 + i * 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    }
    
    return stats_image;
}

void Visualizer::initializeColorPalette() {
    // Generate a diverse color palette for tracks
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(50, 255);
    
    for (int i = 0; i < 100; ++i) {
        cv::Scalar color(dis(gen), dis(gen), dis(gen));
        track_colors_.push_back(color);
    }
}

cv::Scalar Visualizer::getTrackColor(int track_id) {
    if (track_id >= 0 && track_id < static_cast<int>(track_colors_.size())) {
        return track_colors_[track_id];
    }
    return cv::Scalar(255, 255, 255); // White default
}

void Visualizer::drawBoundingBox2D(cv::Mat& image, const cv::Rect2d& bbox,
                                  const cv::Scalar& color, int thickness,
                                  const std::string& label) {
    
    cv::rectangle(image, bbox, color, thickness);
    
    if (!label.empty()) {
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                           params_.text_scale, params_.text_thickness, nullptr);
        cv::Point text_origin(bbox.x, bbox.y - 5);
        
        // Draw text background
        cv::rectangle(image, 
                     cv::Rect(text_origin.x, text_origin.y - text_size.height,
                             text_size.width, text_size.height + 5),
                     color, -1);
        
        cv::putText(image, label, text_origin, cv::FONT_HERSHEY_SIMPLEX,
                   params_.text_scale, cv::Scalar(255, 255, 255), params_.text_thickness);
    }
}

void Visualizer::drawBoundingBox3D(cv::Mat& image, const Eigen::Vector3d& center_3d,
                                  const Eigen::Vector3d& size_3d,
                                  const Eigen::Matrix3d& camera_matrix,
                                  const cv::Scalar& color, int thickness) {
    
    // Define 8 corners of 3D bounding box
    std::vector<Eigen::Vector3d> corners_3d;
    double half_width = size_3d.x() / 2.0;
    double half_length = size_3d.y() / 2.0;
    double half_height = size_3d.z() / 2.0;
    
    corners_3d.push_back(center_3d + Eigen::Vector3d(-half_width, -half_length, -half_height));
    corners_3d.push_back(center_3d + Eigen::Vector3d(half_width, -half_length, -half_height));
    corners_3d.push_back(center_3d + Eigen::Vector3d(half_width, half_length, -half_height));
    corners_3d.push_back(center_3d + Eigen::Vector3d(-half_width, half_length, -half_height));
    corners_3d.push_back(center_3d + Eigen::Vector3d(-half_width, -half_length, half_height));
    corners_3d.push_back(center_3d + Eigen::Vector3d(half_width, -half_length, half_height));
    corners_3d.push_back(center_3d + Eigen::Vector3d(half_width, half_length, half_height));
    corners_3d.push_back(center_3d + Eigen::Vector3d(-half_width, half_length, half_height));
    
    // Project to 2D
    std::vector<cv::Point2f> corners_2d;
    for (const auto& corner_3d : corners_3d) {
        corners_2d.push_back(project3DTo2D(corner_3d, camera_matrix));
    }
    
    // Draw bounding box edges
    std::vector<std::pair<int, int>> edges = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Bottom face
        {4, 5}, {5, 6}, {6, 7}, {7, 4}, // Top face
        {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Vertical edges
    };
    
    for (const auto& edge : edges) {
        cv::Point2f p1 = corners_2d[edge.first];
        cv::Point2f p2 = corners_2d[edge.second];
        
        // Check if points are within image bounds
        if (p1.x >= 0 && p1.y >= 0 && p1.x < image.cols && p1.y < image.rows &&
            p2.x >= 0 && p2.y >= 0 && p2.x < image.cols && p2.y < image.rows) {
            cv::line(image, p1, p2, color, thickness);
        }
    }
}

void Visualizer::drawVelocityVector(cv::Mat& image, const Eigen::Vector3d& position,
                                   const Eigen::Vector3d& velocity,
                                   const Eigen::Matrix3d& camera_matrix,
                                   const cv::Scalar& color, double scale) {
    
    cv::Point2f start = project3DTo2D(position, camera_matrix);
    cv::Point2f end = project3DTo2D(position + velocity * scale, camera_matrix);
    
    if (start.x >= 0 && start.y >= 0 && start.x < image.cols && start.y < image.rows &&
        end.x >= 0 && end.y >= 0 && end.x < image.cols && end.y < image.rows) {
        
        cv::arrowedLine(image, start, end, color, 2);
    }
}

void Visualizer::drawUncertaintyEllipse(cv::Mat& image, const cv::Point2f& center,
                                       const Eigen::Matrix2d& covariance,
                                       const cv::Scalar& color, double confidence) {
    
    // Calculate eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(covariance);
    Eigen::Vector2d eigenvalues = solver.eigenvalues();
    Eigen::Matrix2d eigenvectors = solver.eigenvectors();
    
    // Calculate ellipse parameters
    double chi_squared = 5.991; // 95% confidence for 2D
    double a = std::sqrt(chi_squared * eigenvalues(1));
    double b = std::sqrt(chi_squared * eigenvalues(0));
    double angle = std::atan2(eigenvectors(1, 1), eigenvectors(0, 1)) * 180.0 / M_PI;
    
    cv::ellipse(image, center, cv::Size(a, b), angle, 0, 360, color, 1);
}

cv::Point2f Visualizer::project3DTo2D(const Eigen::Vector3d& point_3d,
                                      const Eigen::Matrix3d& camera_matrix) {
    
    if (std::abs(point_3d.z()) > 1e-6) {
        double u = camera_matrix(0, 0) * point_3d.x() / point_3d.z() + camera_matrix(0, 2);
        double v = camera_matrix(1, 1) * point_3d.y() / point_3d.z() + camera_matrix(1, 2);
        return cv::Point2f(u, v);
    }
    
    return cv::Point2f(-1, -1);
}

void Visualizer::add3DBoundingBox(const Eigen::Vector3d& center_3d,
                                 const Eigen::Vector3d& size_3d,
                                 const std::string& id,
                                 const std::array<double, 3>& color) {
#ifndef WITH_PCL
    (void)center_3d; (void)size_3d; (void)id; (void)color;
    return;
#else
    if (!viewer_3d_) return;

    // Create bounding box as a wireframe cube
    double half_width = size_3d.x() / 2.0;
    double half_length = size_3d.y() / 2.0;
    double half_height = size_3d.z() / 2.0;

    pcl::PointXYZ min_pt(center_3d.x() - half_width,
                         center_3d.y() - half_length,
                         center_3d.z() - half_height);
    pcl::PointXYZ max_pt(center_3d.x() + half_width,
                         center_3d.y() + half_length,
                         center_3d.z() + half_height);

    viewer_3d_->addCube(min_pt.x, max_pt.x, min_pt.y, max_pt.y, min_pt.z, max_pt.z,
                       color[0], color[1], color[2], id);
    viewer_3d_->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                           pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id);
#endif
}

std::array<double, 3> Visualizer::bgrToRgb(const cv::Scalar& bgr_color) {
    return {bgr_color[2] / 255.0, bgr_color[1] / 255.0, bgr_color[0] / 255.0};
}

std::string Visualizer::getClassName(int class_id) {
    switch (class_id) {
        case 0: return "Car";
        case 1: return "Pedestrian";
        case 2: return "Cyclist";
        default: return "Unknown";
    }
}

// PerformanceProfiler Implementation
PerformanceProfiler::PerformanceProfiler() {
}

void PerformanceProfiler::startTimer(const std::string& operation_name) {
    timers_[operation_name].start_time = std::chrono::high_resolution_clock::now();
    timers_[operation_name].is_running = true;
}

void PerformanceProfiler::stopTimer(const std::string& operation_name) {
    auto it = timers_.find(operation_name);
    if (it != timers_.end() && it->second.is_running) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - it->second.start_time).count();
        
        it->second.execution_times.push_back(duration / 1000.0); // Convert to milliseconds
        it->second.is_running = false;
    }
}

double PerformanceProfiler::getAverageTime(const std::string& operation_name) const {
    auto it = timers_.find(operation_name);
    if (it != timers_.end() && !it->second.execution_times.empty()) {
        double sum = 0.0;
        for (double time : it->second.execution_times) {
            sum += time;
        }
        return sum / it->second.execution_times.size();
    }
    return 0.0;
}

void PerformanceProfiler::printSummary() const {
    std::cout << "\n=== Performance Summary ===" << std::endl;
    for (const auto& [name, data] : timers_) {
        if (!data.execution_times.empty()) {
            double avg_time = getAverageTime(name);
            std::cout << name << ": " << std::fixed << std::setprecision(2) 
                     << avg_time << " ms (avg), " 
                     << data.execution_times.size() << " samples" << std::endl;
        }
    }
    std::cout << "=========================" << std::endl;
}

void PerformanceProfiler::reset() {
    timers_.clear();
}

} // namespace tracking
