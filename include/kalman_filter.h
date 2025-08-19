#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace tracking {

/**
 * @brief Extended Kalman Filter implementation for 3D object tracking
 * 
 * State vector: [px, py, pz, vx, vy, vz, ax, ay, az]
 * - Position (px, py, pz)
 * - Velocity (vx, vy, vz)  
 * - Acceleration (ax, ay, az)
 */
class ExtendedKalmanFilter {
public:
    ExtendedKalmanFilter();
    ~ExtendedKalmanFilter() = default;

    /**
     * @brief Initialize the filter with initial state
     * @param initial_state Initial state vector [px, py, pz, vx, vy, vz, ax, ay, az]
     */
    void initialize(const Eigen::VectorXd& initial_state);

    /**
     * @brief Predict the next state using motion model
     * @param dt Time step in seconds
     */
    void predict(double dt);

    /**
     * @brief Update state with camera measurement (2D bounding box)
     * @param measurement 2D measurement [center_x, center_y, width, height]
     * @param camera_matrix Camera intrinsic matrix
     * @param R_camera Measurement noise covariance for camera
     */
    void updateCamera(const Eigen::VectorXd& measurement, 
                     const Eigen::Matrix3d& camera_matrix,
                     const Eigen::MatrixXd& R_camera);

    /**
     * @brief Update state with LiDAR measurement (3D bounding box)
     * @param measurement 3D measurement [center_x, center_y, center_z, width, height, depth]
     * @param R_lidar Measurement noise covariance for LiDAR
     */
    void updateLiDAR(const Eigen::VectorXd& measurement, 
                    const Eigen::MatrixXd& R_lidar);

    /**
     * @brief Get current state estimate
     * @return Current state vector
     */
    Eigen::VectorXd getState() const { return x_; }

    /**
     * @brief Get current covariance matrix
     * @return Current covariance matrix
     */
    Eigen::MatrixXd getCovariance() const { return P_; }

    /**
     * @brief Check if filter is initialized
     * @return True if initialized
     */
    bool isInitialized() const { return is_initialized_; }

private:
    // State vector [px, py, pz, vx, vy, vz, ax, ay, az]
    Eigen::VectorXd x_;
    
    // Covariance matrix
    Eigen::MatrixXd P_;
    
    // Process noise covariance
    Eigen::MatrixXd Q_;
    
    // State transition matrix
    Eigen::MatrixXd F_;
    
    // Initialization flag
    bool is_initialized_;
    
    // Previous timestamp
    double previous_timestamp_;

    /**
     * @brief Calculate Jacobian matrix for camera measurements
     * @param state Current state vector
     * @param camera_matrix Camera intrinsic matrix
     * @return Jacobian matrix
     */
    Eigen::MatrixXd calculateCameraJacobian(const Eigen::VectorXd& state,
                                           const Eigen::Matrix3d& camera_matrix);

    /**
     * @brief Calculate Jacobian matrix for LiDAR measurements
     * @param state Current state vector
     * @return Jacobian matrix
     */
    Eigen::MatrixXd calculateLiDARJacobian(const Eigen::VectorXd& state);

    /**
     * @brief Project 3D state to 2D camera coordinates
     * @param state Current state vector
     * @param camera_matrix Camera intrinsic matrix
     * @return Projected 2D coordinates
     */
    Eigen::VectorXd projectToCamera(const Eigen::VectorXd& state,
                                   const Eigen::Matrix3d& camera_matrix);

    /**
     * @brief Update process noise matrix Q based on time step
     * @param dt Time step in seconds
     */
    void updateProcessNoise(double dt);
};

} // namespace tracking
