#pragma once

#include "object_tracker.h"
#include "types.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

namespace tracking {

/**
 * @brief Camera-based object detection using deep learning
 */
class CameraDetector {
public:
    CameraDetector();
    ~CameraDetector() = default;

    /**
     * @brief Initialize the detector with model files
     * @param model_path Path to ONNX/TensorFlow model
     * @param config_path Path to configuration file
     * @param class_names_path Path to class names file
     * @return True if successful
     */
    bool initialize(const std::string& model_path,
                   const std::string& config_path = "",
                   const std::string& class_names_path = "");

    /**
     * @brief Detect objects in image
     * @param image Input image
     * @param confidence_threshold Minimum confidence threshold
     * @param nms_threshold Non-maximum suppression threshold
     * @return Vector of 2D detections
     */
    std::vector<Detection> detect(const cv::Mat& image,
                                 double confidence_threshold = 0.5,
                                 double nms_threshold = 0.4);

    /**
     * @brief Set input image size for the network
     * @param width Input width
     * @param height Input height
     */
    void setInputSize(int width, int height);

private:
    cv::dnn::Net net_;
    std::vector<std::string> class_names_;
    cv::Size input_size_;
    bool is_initialized_;

    /**
     * @brief Preprocess image for neural network
     * @param image Input image
     * @return Preprocessed blob
     */
    cv::Mat preprocessImage(const cv::Mat& image);

    /**
     * @brief Post-process network outputs
     * @param outputs Network outputs
     * @param image_size Original image size
     * @param confidence_threshold Confidence threshold
     * @param nms_threshold NMS threshold
     * @return Vector of detections
     */
    std::vector<Detection> postprocessOutputs(const std::vector<cv::Mat>& outputs,
                                             const cv::Size& image_size,
                                             double confidence_threshold,
                                             double nms_threshold);
};

/**
 * @brief LiDAR-based object detection using clustering
 */
class LiDARDetector {
public:
    LiDARDetector();
    ~LiDARDetector() = default;

    /**
     * @brief Initialize the detector with parameters
     * @param voxel_size Voxel grid filter size
     * @param cluster_tolerance Clustering tolerance
     * @param min_cluster_size Minimum cluster size
     * @param max_cluster_size Maximum cluster size
     */
    void initialize(double voxel_size = 0.1,
                   double cluster_tolerance = 0.5,
                   int min_cluster_size = 10,
                   int max_cluster_size = 2500);

    /**
     * @brief Detect objects in point cloud
     * @param cloud Input point cloud
     * @param crop_min Minimum crop box coordinates
     * @param crop_max Maximum crop box coordinates
     * @return Vector of 3D detections
     */
    std::vector<Detection> detect(const PointCloudPtr& cloud,
                                 const Eigen::Vector3f& crop_min = Eigen::Vector3f(-50, -50, -3),
                                 const Eigen::Vector3f& crop_max = Eigen::Vector3f(50, 50, 3));

private:
    double voxel_size_;
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
    
#ifdef WITH_PCL
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter_;
    pcl::CropBox<pcl::PointXYZI> crop_filter_;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> cluster_extractor_;
#endif

    /**
     * @brief Preprocess point cloud (filtering, cropping)
     * @param cloud Input point cloud
     * @param crop_min Minimum crop coordinates
     * @param crop_max Maximum crop coordinates
     * @return Filtered point cloud
     */
    PointCloudPtr preprocessPointCloud(const PointCloudPtr& cloud,
        const Eigen::Vector3f& crop_min,
        const Eigen::Vector3f& crop_max);

    /**
     * @brief Extract clusters from point cloud
     * @param cloud Preprocessed point cloud
     * @return Vector of cluster indices
     */
#ifdef WITH_PCL
    std::vector<pcl::PointIndices> extractClusters(const PointCloudPtr& cloud);
#else
    std::vector<int> extractClusters(const PointCloudPtr& /*cloud*/) { return {}; }
#endif

    /**
     * @brief Calculate bounding box for cluster
     * @param cloud Point cloud
     * @param indices Cluster indices
     * @return Detection with 3D bounding box
     */
    Detection calculateBoundingBox(const PointCloudPtr& cloud,
                                  const /*indices placeholder*/ void* indices);

    /**
     * @brief Classify cluster based on geometric features
     * @param detection Detection with bounding box
     * @return Classified object type
     */
    int classifyCluster(const Detection& detection);
};

/**
 * @brief Multimodal perception module combining camera and LiDAR
 */
class PerceptionModule {
public:
    PerceptionModule();
    ~PerceptionModule() = default;

    /**
     * @brief Initialize perception module
     * @param camera_model_path Path to camera detection model
     * @param camera_config_path Path to camera config
     * @param class_names_path Path to class names
     * @return True if successful
     */
    bool initialize(const std::string& camera_model_path,
                   const std::string& camera_config_path = "",
                   const std::string& class_names_path = "");

    /**
     * @brief Process multimodal sensor data
     * @param image Camera image
     * @param cloud LiDAR point cloud
     * @param camera_matrix Camera intrinsic matrix
     * @param camera_to_lidar_transform Transformation matrix
     * @param timestamp Current timestamp
     * @return Fused detections
     */
    std::vector<Detection> processFrame(const cv::Mat& image,
                                       const PointCloudPtr& cloud,
                                       const Eigen::Matrix3d& camera_matrix,
                                       const Eigen::Matrix4d& camera_to_lidar_transform,
                                       double timestamp);

private:
    CameraDetector camera_detector_;
    LiDARDetector lidar_detector_;

    /**
     * @brief Fuse camera and LiDAR detections
     * @param camera_detections 2D camera detections
     * @param lidar_detections 3D LiDAR detections
     * @param camera_matrix Camera intrinsic matrix
     * @param camera_to_lidar_transform Transformation matrix
     * @return Fused detections
     */
    std::vector<Detection> fuseDetections(const std::vector<Detection>& camera_detections,
                                         const std::vector<Detection>& lidar_detections,
                                         const Eigen::Matrix3d& camera_matrix,
                                         const Eigen::Matrix4d& camera_to_lidar_transform);

    /**
     * @brief Project 3D point to 2D image coordinates
     * @param point_3d 3D point in LiDAR coordinates
     * @param camera_matrix Camera intrinsic matrix
     * @param transform Transformation matrix
     * @return 2D image coordinates
     */
    cv::Point2f project3DTo2D(const Eigen::Vector3d& point_3d,
                              const Eigen::Matrix3d& camera_matrix,
                              const Eigen::Matrix4d& transform);

    /**
     * @brief Calculate IoU between two 2D bounding boxes
     * @param box1 First bounding box
     * @param box2 Second bounding box
     * @return IoU value [0, 1]
     */
    double calculateIoU(const cv::Rect2d& box1, const cv::Rect2d& box2);
};

} // namespace tracking
