#pragma once

#include <opencv2/opencv.hpp>
#include "types.h"
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace tracking {

/**
 * @brief Structure to hold IMU/GPS data
 */
struct IMUData {
    double timestamp;
    Eigen::Vector3d linear_acceleration;  // m/s^2
    Eigen::Vector3d angular_velocity;     // rad/s
    Eigen::Vector3d position;             // GPS position (lat, lon, alt)
    Eigen::Vector3d velocity;             // m/s
    
    IMUData() : timestamp(0.0) {
        linear_acceleration.setZero();
        angular_velocity.setZero();
        position.setZero();
        velocity.setZero();
    }
};

/**
 * @brief KITTI dataset loader for multimodal sensor data
 */
class KITTIDataLoader {
public:
    KITTIDataLoader(const std::string& dataset_path);
    ~KITTIDataLoader() = default;

    /**
     * @brief Initialize the data loader
     * @return True if successful
     */
    bool initialize();

    /**
     * @brief Load camera image at given index
     * @param camera_id Camera identifier (0-3)
     * @param frame_index Frame index
     * @return OpenCV image
     */
    cv::Mat loadCameraImage(int camera_id, int frame_index);

    /**
     * @brief Load LiDAR point cloud at given index
     * @param frame_index Frame index
     * @return Point cloud (PCL or simplified)
     */
    PointCloudPtr loadLiDARPointCloud(int frame_index);

    /**
     * @brief Load IMU/GPS data at given index
     * @param frame_index Frame index
     * @return IMU data structure
     */
    IMUData loadIMUData(int frame_index);

    /**
     * @brief Get camera calibration matrix
     * @param camera_id Camera identifier (0-3)
     * @return 3x3 camera intrinsic matrix
     */
    Eigen::Matrix3d getCameraMatrix(int camera_id) const;

    /**
     * @brief Get camera to LiDAR transformation matrix
     * @param camera_id Camera identifier (0-3)
     * @return 4x4 transformation matrix
     */
    Eigen::Matrix4d getCameraToLiDARTransform(int camera_id) const;

    /**
     * @brief Get timestamp for given frame
     * @param frame_index Frame index
     * @return Timestamp in seconds
     */
    double getTimestamp(int frame_index) const;

    /**
     * @brief Get total number of frames
     * @return Number of frames
     */
    int getNumFrames() const { return num_frames_; }

    /**
     * @brief Check if frame index is valid
     * @param frame_index Frame index to check
     * @return True if valid
     */
    bool isValidFrame(int frame_index) const;

private:
    std::string dataset_path_;
    int num_frames_;
    
    // Camera calibration data
    std::vector<Eigen::Matrix3d> camera_matrices_;
    std::vector<Eigen::Matrix4d> camera_to_lidar_transforms_;
    
    // Timestamps
    std::vector<double> timestamps_;
    std::vector<double> velodyne_timestamps_;
    std::vector<double> oxts_timestamps_;

    /**
     * @brief Load camera calibration data
     * @return True if successful
     */
    bool loadCalibrationData();

    /**
     * @brief Load timestamp files
     * @return True if successful
     */
    bool loadTimestamps();

    /**
     * @brief Parse calibration file
     * @param filepath Path to calibration file
     * @return True if successful
     */
    bool parseCalibrationFile(const std::string& filepath);

    /**
     * @brief Parse timestamp file
     * @param filepath Path to timestamp file
     * @param timestamps Output vector for timestamps
     * @return True if successful
     */
    bool parseTimestampFile(const std::string& filepath, std::vector<double>& timestamps);

    /**
     * @brief Convert timestamp string to seconds
     * @param timestamp_str Timestamp string in KITTI format
     * @return Timestamp in seconds
     */
    double parseTimestamp(const std::string& timestamp_str);
};

} // namespace tracking
