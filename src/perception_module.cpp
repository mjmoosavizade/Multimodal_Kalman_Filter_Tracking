#include "perception_module.h"
#include <iostream>
#include <algorithm>
#ifdef WITH_PCL
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#endif

namespace tracking {

// CameraDetector Implementation
CameraDetector::CameraDetector() : input_size_(416, 416), is_initialized_(false) {
}

bool CameraDetector::initialize(const std::string& model_path,
                               const std::string& config_path,
                               const std::string& class_names_path) {
    try {
        // Load the neural network
        if (config_path.empty()) {
            net_ = cv::dnn::readNetFromONNX(model_path);
        } else {
            net_ = cv::dnn::readNetFromDarknet(config_path, model_path);
        }
        
        if (net_.empty()) {
            std::cerr << "Failed to load neural network model" << std::endl;
            return false;
        }
        
        // Set backend and target
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        // Load class names if provided
        if (!class_names_path.empty()) {
            std::ifstream file(class_names_path);
            std::string line;
            while (std::getline(file, line)) {
                class_names_.push_back(line);
            }
        } else {
            // Default KITTI classes
            class_names_ = {"Car", "Pedestrian", "Cyclist"};
        }
        
        is_initialized_ = true;
        std::cout << "Camera detector initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing camera detector: " << e.what() << std::endl;
        return false;
    }
}

std::vector<Detection> CameraDetector::detect(const cv::Mat& image,
                                             double confidence_threshold,
                                             double nms_threshold) {
    std::vector<Detection> detections;
    
    if (!is_initialized_ || image.empty()) {
        return detections;
    }
    
    try {
        // Preprocess image
        cv::Mat blob = preprocessImage(image);
        
        // Set input to the network
        net_.setInput(blob);
        
        // Run forward pass
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());
        
        // Post-process outputs
        detections = postprocessOutputs(outputs, image.size(), 
                                      confidence_threshold, nms_threshold);
        
    } catch (const std::exception& e) {
        std::cerr << "Error during camera detection: " << e.what() << std::endl;
    }
    
    return detections;
}

void CameraDetector::setInputSize(int width, int height) {
    input_size_ = cv::Size(width, height);
}

cv::Mat CameraDetector::preprocessImage(const cv::Mat& image) {
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255.0, input_size_, 
                          cv::Scalar(0, 0, 0), true, false, CV_32F);
    return blob;
}

std::vector<Detection> CameraDetector::postprocessOutputs(
    const std::vector<cv::Mat>& outputs,
    const cv::Size& image_size,
    double confidence_threshold,
    double nms_threshold) {
    
    std::vector<Detection> detections;
    std::vector<cv::Rect2d> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    
    // Parse outputs (assuming YOLO format)
    for (const auto& output : outputs) {
        for (int i = 0; i < output.rows; ++i) {
            const float* data = output.ptr<float>(i);
            
            // Extract confidence and class scores
            float confidence = data[4];
            if (confidence < confidence_threshold) continue;
            
            // Find best class
            cv::Point class_id_point;
            double class_confidence;
            cv::minMaxLoc(output.row(i).colRange(5, output.cols), 
                         nullptr, &class_confidence, nullptr, &class_id_point);
            
            if (class_confidence < confidence_threshold) continue;
            
            // Extract bounding box
            float center_x = data[0] * image_size.width;
            float center_y = data[1] * image_size.height;
            float width = data[2] * image_size.width;
            float height = data[3] * image_size.height;
            
            cv::Rect2d box(center_x - width/2, center_y - height/2, width, height);
            
            boxes.push_back(box);
            confidences.push_back(confidence * class_confidence);
            class_ids.push_back(class_id_point.x);
        }
    }
    
    // Apply Non-Maximum Suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, indices);
    
    // Create detection objects
    for (int idx : indices) {
        Detection detection;
        detection.bbox_2d = boxes[idx];
        detection.confidence = confidences[idx];
        detection.class_id = class_ids[idx];
        detections.push_back(detection);
    }
    
    return detections;
}

// LiDARDetector Implementation
#ifdef WITH_PCL
LiDARDetector::LiDARDetector() 
    : voxel_size_(0.1), cluster_tolerance_(0.5), 
      min_cluster_size_(10), max_cluster_size_(2500) {
}

void LiDARDetector::initialize(double voxel_size, double cluster_tolerance,
                              int min_cluster_size, int max_cluster_size) {
    voxel_size_ = voxel_size;
    cluster_tolerance_ = cluster_tolerance;
    min_cluster_size_ = min_cluster_size;
    max_cluster_size_ = max_cluster_size;
    
    // Configure filters
    voxel_filter_.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    cluster_extractor_.setClusterTolerance(cluster_tolerance_);
    cluster_extractor_.setMinClusterSize(min_cluster_size_);
    cluster_extractor_.setMaxClusterSize(max_cluster_size_);
    
    std::cout << "LiDAR detector initialized successfully" << std::endl;
}

std::vector<Detection> LiDARDetector::detect(
    const PointCloudPtr& cloud,
    const Eigen::Vector3f& crop_min,
    const Eigen::Vector3f& crop_max) {
    
    std::vector<Detection> detections;
    
    if (!cloud || cloud->empty()) {
        return detections;
    }
    
    try {
        // Preprocess point cloud
        auto filtered_cloud = preprocessPointCloud(cloud, crop_min, crop_max);
        
        if (filtered_cloud->empty()) {
            return detections;
        }
        
        // Extract clusters
        auto cluster_indices = extractClusters(filtered_cloud);
        
        // Create detections from clusters
        for (const auto& indices : cluster_indices) {
            Detection detection = calculateBoundingBox(filtered_cloud, indices);
            detection.class_id = classifyCluster(detection);
            detections.push_back(detection);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error during LiDAR detection: " << e.what() << std::endl;
    }
    
    return detections;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr LiDARDetector::preprocessPointCloud(
    const PointCloudPtr& cloud,
    const Eigen::Vector3f& crop_min,
    const Eigen::Vector3f& crop_max) {
    
    auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    
    // Crop box filtering
    crop_filter_.setMin(Eigen::Vector4f(crop_min.x(), crop_min.y(), crop_min.z(), 1.0));
    crop_filter_.setMax(Eigen::Vector4f(crop_max.x(), crop_max.y(), crop_max.z(), 1.0));
    crop_filter_.setInputCloud(cloud);
    crop_filter_.filter(*filtered_cloud);
    
    // Voxel grid filtering
    auto voxel_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    voxel_filter_.setInputCloud(filtered_cloud);
    voxel_filter_.filter(*voxel_filtered);
    
    return voxel_filtered;
}

std::vector<pcl::PointIndices> LiDARDetector::extractClusters(
    const PointCloudPtr& cloud) {
    
    // Create KdTree for search
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud);
    
    // Extract clusters
    std::vector<pcl::PointIndices> cluster_indices;
    cluster_extractor_.setSearchMethod(tree);
    cluster_extractor_.setInputCloud(cloud);
    cluster_extractor_.extract(cluster_indices);
    
    return cluster_indices;
}

Detection LiDARDetector::calculateBoundingBox(
    const PointCloudPtr& cloud,
    const pcl::PointIndices& indices) {
    
    Detection detection;
    
    if (indices.indices.empty()) {
        return detection;
    }
    
    // Extract cluster points
    pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(std::make_shared<pcl::PointIndices>(indices));
    extract.filter(*cluster);
    
    // Calculate bounding box
    pcl::PointXYZI min_pt, max_pt;
    pcl::getMinMax3D(*cluster, min_pt, max_pt);
    
    // Set 3D center and size
    detection.center_3d << (min_pt.x + max_pt.x) / 2.0,
                          (min_pt.y + max_pt.y) / 2.0,
                          (min_pt.z + max_pt.z) / 2.0;
    
    detection.size_3d << max_pt.x - min_pt.x,
                        max_pt.y - min_pt.y,
                        max_pt.z - min_pt.z;
    
    detection.confidence = 0.8; // Default confidence for LiDAR detections
    
    return detection;
}
#else
// Fallback implementations when PCL is not available
LiDARDetector::LiDARDetector() : voxel_size_(0.1), cluster_tolerance_(0.5),
    min_cluster_size_(10), max_cluster_size_(2500) {}

void LiDARDetector::initialize(double, double, int, int) {
    std::cout << "LiDAR detector running in fallback mode (no PCL)" << std::endl;
}

std::vector<Detection> LiDARDetector::detect(const PointCloudPtr& /*cloud*/, 
                                             const Eigen::Vector3f& /*crop_min*/, 
                                             const Eigen::Vector3f& /*crop_max*/) {
    // Without PCL, we cannot perform clustering. Return empty.
    return {};
}
#endif

int LiDARDetector::classifyCluster(const Detection& detection) {
    // Simple geometric classification based on size
    double width = detection.size_3d.x();
    double height = detection.size_3d.z();
    double length = detection.size_3d.y();
    
    // Car: typically 4-5m long, 1.5-2m wide, 1.5-2m high
    if (length > 3.0 && length < 6.0 && width > 1.0 && width < 2.5 && height > 1.0 && height < 2.5) {
        return 0; // Car
    }
    // Pedestrian: typically 0.5-1m wide, 1.5-2m high
    else if (width < 1.0 && height > 1.2 && height < 2.2) {
        return 1; // Pedestrian
    }
    // Cyclist: similar to pedestrian but potentially wider
    else if (width < 1.5 && height > 1.0 && height < 2.0) {
        return 2; // Cyclist
    }
    
    return 0; // Default to car
}

// PerceptionModule Implementation
PerceptionModule::PerceptionModule() {
}

bool PerceptionModule::initialize(const std::string& camera_model_path,
                                 const std::string& camera_config_path,
                                 const std::string& class_names_path) {
    
    // Initialize camera detector
    if (!camera_detector_.initialize(camera_model_path, camera_config_path, class_names_path)) {
        std::cerr << "Failed to initialize camera detector" << std::endl;
        return false;
    }
    
    // Initialize LiDAR detector
    lidar_detector_.initialize();
    
    std::cout << "Perception module initialized successfully" << std::endl;
    return true;
}

std::vector<Detection> PerceptionModule::processFrame(
    const cv::Mat& image,
    const PointCloudPtr& cloud,
     const Eigen::Matrix3d& camera_matrix,
     const Eigen::Matrix4d& camera_to_lidar_transform,
     double timestamp) {
    
    // Detect objects in camera image
    auto camera_detections = camera_detector_.detect(image);
    
    // Detect objects in LiDAR point cloud
    auto lidar_detections = lidar_detector_.detect(cloud);
    
    // Set timestamps
    for (auto& detection : camera_detections) {
        detection.timestamp = timestamp;
    }
    for (auto& detection : lidar_detections) {
        detection.timestamp = timestamp;
    }
    
    // Fuse detections
    auto fused_detections = fuseDetections(camera_detections, lidar_detections,
                                          camera_matrix, camera_to_lidar_transform);
    
    return fused_detections;
}

std::vector<Detection> PerceptionModule::fuseDetections(
    const std::vector<Detection>& camera_detections,
    const std::vector<Detection>& lidar_detections,
    const Eigen::Matrix3d& camera_matrix,
    const Eigen::Matrix4d& camera_to_lidar_transform) {
    
    std::vector<Detection> fused_detections;
    std::vector<bool> lidar_used(lidar_detections.size(), false);
    
    // For each camera detection, try to find corresponding LiDAR detection
    for (const auto& cam_det : camera_detections) {
        Detection fused_det = cam_det;
        double best_iou = 0.0;
        int best_lidar_idx = -1;
        
        for (size_t i = 0; i < lidar_detections.size(); ++i) {
            if (lidar_used[i]) continue;
            
            const auto& lidar_det = lidar_detections[i];
            
            // Project LiDAR detection to camera coordinates
            cv::Point2f projected_center = project3DTo2D(lidar_det.center_3d,
                                                        camera_matrix,
                                                        camera_to_lidar_transform);
            
            // Create approximate 2D bounding box for LiDAR detection
            cv::Rect2d lidar_bbox_2d(projected_center.x - 25, projected_center.y - 25, 50, 50);
            
            // Calculate IoU
            double iou = calculateIoU(cam_det.bbox_2d, lidar_bbox_2d);
            
            if (iou > best_iou && iou > 0.1) {
                best_iou = iou;
                best_lidar_idx = i;
            }
        }
        
        // Fuse with best matching LiDAR detection
        if (best_lidar_idx >= 0) {
            const auto& lidar_det = lidar_detections[best_lidar_idx];
            fused_det.center_3d = lidar_det.center_3d;
            fused_det.size_3d = lidar_det.size_3d;
            fused_det.confidence = (cam_det.confidence + lidar_det.confidence) / 2.0;
            lidar_used[best_lidar_idx] = true;
        }
        
        fused_detections.push_back(fused_det);
    }
    
    // Add unmatched LiDAR detections
    for (size_t i = 0; i < lidar_detections.size(); ++i) {
        if (!lidar_used[i]) {
            fused_detections.push_back(lidar_detections[i]);
        }
    }
    
    return fused_detections;
}

cv::Point2f PerceptionModule::project3DTo2D(const Eigen::Vector3d& point_3d,
                                           const Eigen::Matrix3d& camera_matrix,
                                           const Eigen::Matrix4d& transform) {
    
    // Transform point from LiDAR to camera coordinates
    Eigen::Vector4d point_homo(point_3d.x(), point_3d.y(), point_3d.z(), 1.0);
    Eigen::Vector4d point_cam = transform * point_homo;
    
    // Project to image coordinates
    if (std::abs(point_cam.z()) > 1e-6) {
        double u = camera_matrix(0, 0) * point_cam.x() / point_cam.z() + camera_matrix(0, 2);
        double v = camera_matrix(1, 1) * point_cam.y() / point_cam.z() + camera_matrix(1, 2);
        return cv::Point2f(u, v);
    }
    
    return cv::Point2f(0, 0);
}

double PerceptionModule::calculateIoU(const cv::Rect2d& box1, const cv::Rect2d& box2) {
    cv::Rect2d intersection = box1 & box2;
    double intersection_area = intersection.area();
    double union_area = box1.area() + box2.area() - intersection_area;
    
    return (union_area > 0) ? intersection_area / union_area : 0.0;
}

} // namespace tracking
