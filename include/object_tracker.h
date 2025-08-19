#pragma once

#include "kalman_filter.h"
#include <opencv2/opencv.hpp>
#include "types.h"
#include <vector>
#include <unordered_map>
#include <memory>

namespace tracking {

/**
 * @brief Structure to represent a detected object
 */
struct Detection {
    int class_id;           // Object class (0: car, 1: pedestrian, 2: cyclist)
    double confidence;      // Detection confidence [0, 1]
    cv::Rect2d bbox_2d;     // 2D bounding box in image coordinates
    Eigen::Vector3d center_3d;  // 3D center position
    Eigen::Vector3d size_3d;     // 3D size (width, height, depth)
    double timestamp;       // Detection timestamp
    
    Detection() : class_id(-1), confidence(0.0), timestamp(0.0) {}
};

/**
 * @brief Structure to represent a tracked object
 */
struct TrackedObject {
    int id;                                    // Unique track ID
    int class_id;                             // Object class
    std::unique_ptr<ExtendedKalmanFilter> ekf; // Kalman filter
    double last_update_time;                  // Last update timestamp
    int consecutive_misses;                   // Number of consecutive missed detections
    double confidence;                        // Track confidence
    
    TrackedObject(int track_id, int obj_class) 
        : id(track_id), class_id(obj_class), ekf(std::make_unique<ExtendedKalmanFilter>()),
          last_update_time(0.0), consecutive_misses(0), confidence(1.0) {}
};

/**
 * @brief Multi-object tracker using Kalman filters
 */
class MultiObjectTracker {
public:
    MultiObjectTracker();
    ~MultiObjectTracker() = default;

    /**
     * @brief Update tracker with new detections
     * @param detections Vector of detected objects
     * @param camera_matrix Camera intrinsic matrix
     */
    void update(const std::vector<Detection>& detections,
                const Eigen::Matrix3d& camera_matrix);

    /**
     * @brief Get all active tracks
     * @return Vector of tracked objects
     */
    std::vector<TrackedObject*> getActiveTracks();

    /**
     * @brief Get track by ID
     * @param track_id Track identifier
     * @return Pointer to tracked object or nullptr if not found
     */
    TrackedObject* getTrack(int track_id);

    /**
     * @brief Set maximum number of consecutive misses before track deletion
     * @param max_misses Maximum consecutive misses
     */
    void setMaxConsecutiveMisses(int max_misses) { max_consecutive_misses_ = max_misses; }

    /**
     * @brief Set association distance threshold
     * @param threshold Distance threshold for data association
     */
    void setAssociationThreshold(double threshold) { association_threshold_ = threshold; }

private:
    std::unordered_map<int, std::unique_ptr<TrackedObject>> tracks_;
    int next_track_id_;
    int max_consecutive_misses_;
    double association_threshold_;
    
    // Measurement noise covariances
    Eigen::MatrixXd R_camera_;
    Eigen::MatrixXd R_lidar_;

    /**
     * @brief Associate detections with existing tracks
     * @param detections Vector of detections
     * @return Vector of pairs (track_id, detection_index), -1 for new tracks
     */
    std::vector<std::pair<int, int>> associateDetections(
        const std::vector<Detection>& detections);

    /**
     * @brief Calculate distance between track prediction and detection
     * @param track Tracked object
     * @param detection Detection
     * @param camera_matrix Camera intrinsic matrix
     * @return Distance metric
     */
    double calculateDistance(const TrackedObject& track, 
                           const Detection& detection,
                           const Eigen::Matrix3d& camera_matrix);

    /**
     * @brief Create new track from detection
     * @param detection Detection to create track from
     * @return Track ID
     */
    int createNewTrack(const Detection& detection);

    /**
     * @brief Remove tracks that haven't been updated for too long
     */
    void removeOldTracks();

    /**
     * @brief Initialize measurement noise covariances
     */
    void initializeNoiseCovarianceMatrices();
};

} // namespace tracking
