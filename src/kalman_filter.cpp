#include "kalman_filter.h"
#include <iostream>
#include <cmath>

namespace tracking {

ExtendedKalmanFilter::ExtendedKalmanFilter() 
    : is_initialized_(false), previous_timestamp_(0.0) {
    
    // Initialize state vector (9D: px, py, pz, vx, vy, vz, ax, ay, az)
    x_ = Eigen::VectorXd::Zero(9);
    
    // Initialize covariance matrix
    P_ = Eigen::MatrixXd::Identity(9, 9);
    P_ *= 1000.0; // Large initial uncertainty
    
    // Initialize state transition matrix F
    F_ = Eigen::MatrixXd::Identity(9, 9);
    
    // Initialize process noise covariance Q
    Q_ = Eigen::MatrixXd::Zero(9, 9);
}

void ExtendedKalmanFilter::initialize(const Eigen::VectorXd& initial_state) {
    if (initial_state.size() != 9) {
        std::cerr << "Error: Initial state must be 9-dimensional" << std::endl;
        return;
    }
    
    x_ = initial_state;
    
    // Set reasonable initial covariance
    P_ = Eigen::MatrixXd::Identity(9, 9);
    P_.block<3, 3>(0, 0) *= 1.0;    // Position uncertainty: 1m
    P_.block<3, 3>(3, 3) *= 10.0;   // Velocity uncertainty: 10 m/s
    P_.block<3, 3>(6, 6) *= 100.0;  // Acceleration uncertainty: 100 m/s²
    
    is_initialized_ = true;
}

void ExtendedKalmanFilter::predict(double dt) {
    if (!is_initialized_) {
        std::cerr << "Error: Filter not initialized" << std::endl;
        return;
    }
    
    // Update state transition matrix F for constant acceleration model
    F_ = Eigen::MatrixXd::Identity(9, 9);
    
    // Position = position + velocity*dt + 0.5*acceleration*dt²
    F_.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
    F_.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * 0.5 * dt * dt;
    
    // Velocity = velocity + acceleration*dt
    F_.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;
    
    // Update process noise matrix
    updateProcessNoise(dt);
    
    // Predict state
    x_ = F_ * x_;
    
    // Predict covariance
    P_ = F_ * P_ * F_.transpose() + Q_;
    
    previous_timestamp_ = previous_timestamp_ + dt;
}

void ExtendedKalmanFilter::updateCamera(const Eigen::VectorXd& measurement,
                                       const Eigen::Matrix3d& camera_matrix,
                                       const Eigen::MatrixXd& R_camera) {
    if (!is_initialized_) {
        std::cerr << "Error: Filter not initialized" << std::endl;
        return;
    }
    
    // Calculate Jacobian matrix
    Eigen::MatrixXd H = calculateCameraJacobian(x_, camera_matrix);
    
    // Calculate predicted measurement
    Eigen::VectorXd z_pred = projectToCamera(x_, camera_matrix);
    
    // Calculate innovation
    Eigen::VectorXd y = measurement - z_pred;
    
    // Calculate innovation covariance
    Eigen::MatrixXd S = H * P_ * H.transpose() + R_camera;
    
    // Calculate Kalman gain
    Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
    
    // Update state
    x_ = x_ + K * y;
    
    // Update covariance
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x_.size(), x_.size());
    P_ = (I - K * H) * P_;
}

void ExtendedKalmanFilter::updateLiDAR(const Eigen::VectorXd& measurement,
                                      const Eigen::MatrixXd& R_lidar) {
    if (!is_initialized_) {
        std::cerr << "Error: Filter not initialized" << std::endl;
        return;
    }
    
    // For LiDAR, we have direct 3D position measurements
    // Measurement model: z = [px, py, pz]
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 9);
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    
    // Calculate innovation
    Eigen::VectorXd z_pred = H * x_;
    Eigen::VectorXd y = measurement - z_pred;
    
    // Calculate innovation covariance
    Eigen::MatrixXd S = H * P_ * H.transpose() + R_lidar;
    
    // Calculate Kalman gain
    Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
    
    // Update state
    x_ = x_ + K * y;
    
    // Update covariance
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x_.size(), x_.size());
    P_ = (I - K * H) * P_;
}

Eigen::MatrixXd ExtendedKalmanFilter::calculateCameraJacobian(
    const Eigen::VectorXd& state, const Eigen::Matrix3d& camera_matrix) {
    
    double px = state(0);
    double py = state(1);
    double pz = state(2);
    
    double fx = camera_matrix(0, 0);
    double fy = camera_matrix(1, 1);
    double cx = camera_matrix(0, 2);
    double cy = camera_matrix(1, 2);
    
    // Jacobian for camera projection model
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 9);
    
    if (std::abs(pz) > 1e-6) {
        // Derivatives of u = fx * px/pz + cx
        H(0, 0) = fx / pz;           // ∂u/∂px
        H(0, 2) = -fx * px / (pz * pz); // ∂u/∂pz
        
        // Derivatives of v = fy * py/pz + cy
        H(1, 1) = fy / pz;           // ∂v/∂py
        H(1, 2) = -fy * py / (pz * pz); // ∂v/∂pz
        
        // For bounding box width and height, we use simplified model
        // assuming constant size in image coordinates
        H(2, 0) = 0.1; // Simplified derivative for width
        H(3, 1) = 0.1; // Simplified derivative for height
    }
    
    return H;
}

Eigen::MatrixXd ExtendedKalmanFilter::calculateLiDARJacobian(const Eigen::VectorXd& state) {
    // For LiDAR, measurement model is linear: z = [px, py, pz]
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 9);
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    return H;
}

Eigen::VectorXd ExtendedKalmanFilter::projectToCamera(const Eigen::VectorXd& state,
                                                     const Eigen::Matrix3d& camera_matrix) {
    double px = state(0);
    double py = state(1);
    double pz = state(2);
    
    double fx = camera_matrix(0, 0);
    double fy = camera_matrix(1, 1);
    double cx = camera_matrix(0, 2);
    double cy = camera_matrix(1, 2);
    
    Eigen::VectorXd z_pred(4);
    
    if (std::abs(pz) > 1e-6) {
        z_pred(0) = fx * px / pz + cx; // u
        z_pred(1) = fy * py / pz + cy; // v
        z_pred(2) = 50.0;              // Simplified width
        z_pred(3) = 30.0;              // Simplified height
    } else {
        z_pred.setZero();
    }
    
    return z_pred;
}

void ExtendedKalmanFilter::updateProcessNoise(double dt) {
    // Process noise for constant acceleration model
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;
    
    // Noise parameters
    double sigma_a = 2.0; // Acceleration noise standard deviation
    
    Q_ = Eigen::MatrixXd::Zero(9, 9);
    
    // Position-position covariance
    Q_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * dt4 / 4.0 * sigma_a * sigma_a;
    
    // Position-velocity covariance
    Q_.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt3 / 2.0 * sigma_a * sigma_a;
    Q_.block<3, 3>(3, 0) = Q_.block<3, 3>(0, 3);
    
    // Position-acceleration covariance
    Q_.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt2 / 2.0 * sigma_a * sigma_a;
    Q_.block<3, 3>(6, 0) = Q_.block<3, 3>(0, 6);
    
    // Velocity-velocity covariance
    Q_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * dt2 * sigma_a * sigma_a;
    
    // Velocity-acceleration covariance
    Q_.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt * sigma_a * sigma_a;
    Q_.block<3, 3>(6, 3) = Q_.block<3, 3>(3, 6);
    
    // Acceleration-acceleration covariance
    Q_.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * sigma_a * sigma_a;
}

} // namespace tracking
