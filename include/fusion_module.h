#pragma once

#include "object_tracker.h"
#include "data_loader.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace tracking {

/**
 * @brief Sensor fusion module for combining multimodal measurements
 */
class FusionModule {
public:
    FusionModule();
    ~FusionModule() = default;

    /**
     * @brief Initialize fusion module
     * @param max_tracks Maximum number of simultaneous tracks
     * @param association_threshold Distance threshold for data association
     * @param max_consecutive_misses Maximum misses before track deletion
     */
    void initialize(int max_tracks = 50,
                   double association_threshold = 5.0,
                   int max_consecutive_misses = 5);

    /**
     * @brief Process new frame with multimodal detections
     * @param detections Vector of fused detections
     * @param imu_data IMU/GPS data for ego motion compensation
     * @param camera_matrix Camera intrinsic matrix
     * @param timestamp Current timestamp
     */
    void processFrame(const std::vector<Detection>& detections,
                     const IMUData& imu_data,
                     const Eigen::Matrix3d& camera_matrix,
                     double timestamp);

    /**
     * @brief Get all active tracks
     * @return Vector of tracked objects
     */
    std::vector<TrackedObject*> getActiveTracks();

    /**
     * @brief Get track predictions for next time step
     * @param dt Time step for prediction
     * @return Vector of predicted states
     */
    std::vector<Eigen::VectorXd> predictTracks(double dt);

    /**
     * @brief Get tracking statistics
     * @return Tracking statistics structure
     */
    struct TrackingStats {
        int total_tracks;
        int active_tracks;
        int new_tracks_this_frame;
        int lost_tracks_this_frame;
        double average_track_age;
        double tracking_fps;
    };
    
    TrackingStats getTrackingStats() const;

    /**
     * @brief Reset all tracks
     */
    void reset();

private:
    std::unique_ptr<MultiObjectTracker> tracker_;
    IMUData previous_imu_data_;
    double previous_timestamp_;
    bool is_initialized_;
    
    // Tracking statistics
    mutable TrackingStats stats_;
    int frame_count_;
    std::chrono::high_resolution_clock::time_point start_time_;

    /**
     * @brief Compensate detections for ego vehicle motion
     * @param detections Input detections
     * @param current_imu Current IMU data
     * @param previous_imu Previous IMU data
     * @param dt Time difference
     * @return Motion-compensated detections
     */
    std::vector<Detection> compensateEgoMotion(const std::vector<Detection>& detections,
                                              const IMUData& current_imu,
                                              const IMUData& previous_imu,
                                              double dt);

    /**
     * @brief Update tracking statistics
     */
    void updateStatistics();

    /**
     * @brief Validate detection quality
     * @param detection Detection to validate
     * @return True if detection is valid
     */
    bool validateDetection(const Detection& detection);
};

/**
 * @brief Advanced fusion techniques for improved tracking
 */
class AdvancedFusion {
public:
    /**
     * @brief Perform track-to-track fusion for multiple sensors
     * @param tracks_sensor1 Tracks from first sensor
     * @param tracks_sensor2 Tracks from second sensor
     * @param association_threshold Distance threshold
     * @return Fused tracks
     */
    static std::vector<TrackedObject*> fuseTrackToTrack(
        const std::vector<TrackedObject*>& tracks_sensor1,
        const std::vector<TrackedObject*>& tracks_sensor2,
        double association_threshold = 3.0);

    /**
     * @brief Adaptive noise estimation based on tracking performance
     * @param track Tracked object
     * @param recent_innovations Recent innovation vectors
     * @return Estimated noise covariance
     */
    static Eigen::MatrixXd estimateAdaptiveNoise(
        const TrackedObject& track,
        const std::vector<Eigen::VectorXd>& recent_innovations);

    /**
     * @brief Interacting Multiple Model (IMM) for motion model adaptation
     * @param track Tracked object
     * @param motion_models Vector of motion models
     * @param dt Time step
     */
    static void applyIMM(TrackedObject& track,
                        const std::vector<Eigen::MatrixXd>& motion_models,
                        double dt);

private:
    /**
     * @brief Calculate Mahalanobis distance between tracks
     * @param track1 First track
     * @param track2 Second track
     * @return Mahalanobis distance
     */
    static double calculateMahalanobisDistance(const TrackedObject& track1,
                                              const TrackedObject& track2);
};

} // namespace tracking
