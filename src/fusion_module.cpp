#include "fusion_module.h"
#include <iostream>
#include <chrono>
#include <algorithm>

namespace tracking {

FusionModule::FusionModule() 
    : tracker_(std::make_unique<MultiObjectTracker>()),
      previous_timestamp_(0.0), is_initialized_(false), frame_count_(0) {
    
    start_time_ = std::chrono::high_resolution_clock::now();
}

void FusionModule::initialize(int max_tracks, double association_threshold, 
                             int max_consecutive_misses) {
    
    tracker_->setAssociationThreshold(association_threshold);
    tracker_->setMaxConsecutiveMisses(max_consecutive_misses);
    
    is_initialized_ = true;
    std::cout << "Fusion module initialized with max_tracks=" << max_tracks 
              << ", association_threshold=" << association_threshold 
              << ", max_consecutive_misses=" << max_consecutive_misses << std::endl;
}

void FusionModule::processFrame(const std::vector<Detection>& detections,
                               const IMUData& imu_data,
                               const Eigen::Matrix3d& camera_matrix,
                               double timestamp) {
    
    if (!is_initialized_) {
        std::cerr << "Fusion module not initialized" << std::endl;
        return;
    }
    
    // Compensate for ego vehicle motion if we have previous IMU data
    std::vector<Detection> compensated_detections = detections;
    if (previous_timestamp_ > 0 && timestamp > previous_timestamp_) {
        double dt = timestamp - previous_timestamp_;
        compensated_detections = compensateEgoMotion(detections, imu_data, 
                                                   previous_imu_data_, dt);
    }
    
    // Validate detections
    std::vector<Detection> valid_detections;
    for (const auto& detection : compensated_detections) {
        if (validateDetection(detection)) {
            valid_detections.push_back(detection);
        }
    }
    
    // Update tracker with valid detections
    tracker_->update(valid_detections, camera_matrix);
    
    // Update statistics
    updateStatistics();
    
    // Store current data for next iteration
    previous_imu_data_ = imu_data;
    previous_timestamp_ = timestamp;
    frame_count_++;
}

std::vector<TrackedObject*> FusionModule::getActiveTracks() {
    return tracker_->getActiveTracks();
}

std::vector<Eigen::VectorXd> FusionModule::predictTracks(double dt) {
    std::vector<Eigen::VectorXd> predictions;
    
    auto active_tracks = getActiveTracks();
    for (auto* track : active_tracks) {
        if (track->ekf->isInitialized()) {
            // Create a copy of the filter for prediction
            ExtendedKalmanFilter temp_filter = *(track->ekf);
            temp_filter.predict(dt);
            predictions.push_back(temp_filter.getState());
        }
    }
    
    return predictions;
}

FusionModule::TrackingStats FusionModule::getTrackingStats() const {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - start_time_).count();
    
    stats_.tracking_fps = (duration > 0) ? (frame_count_ * 1000.0 / duration) : 0.0;
    
    return stats_;
}

void FusionModule::reset() {
    tracker_ = std::make_unique<MultiObjectTracker>();
    previous_timestamp_ = 0.0;
    frame_count_ = 0;
    start_time_ = std::chrono::high_resolution_clock::now();
    
    // Reset statistics
    stats_ = TrackingStats{};
}

std::vector<Detection> FusionModule::compensateEgoMotion(
    const std::vector<Detection>& detections,
    const IMUData& current_imu,
    const IMUData& previous_imu,
    double dt) {
    
    std::vector<Detection> compensated_detections = detections;
    
    if (dt <= 0) {
        return compensated_detections;
    }
    
    // Calculate ego vehicle displacement
    Eigen::Vector3d ego_velocity_change = current_imu.velocity - previous_imu.velocity;
    Eigen::Vector3d ego_displacement = previous_imu.velocity * dt + 
                                      0.5 * current_imu.linear_acceleration * dt * dt;
    
    // Compensate each detection
    for (auto& detection : compensated_detections) {
        if (detection.center_3d.norm() > 0) {
            // Subtract ego motion from object position
            detection.center_3d -= ego_displacement;
        }
    }
    
    return compensated_detections;
}

void FusionModule::updateStatistics() {
    auto active_tracks = getActiveTracks();
    
    stats_.active_tracks = active_tracks.size();
    stats_.total_tracks = tracker_->getActiveTracks().size(); // This includes all tracks
    
    // Calculate average track age (simplified)
    double total_age = 0.0;
    int valid_tracks = 0;
    
    for (auto* track : active_tracks) {
        if (track->last_update_time > 0) {
            double age = previous_timestamp_ - track->last_update_time;
            total_age += age;
            valid_tracks++;
        }
    }
    
    stats_.average_track_age = (valid_tracks > 0) ? total_age / valid_tracks : 0.0;
    
    // These would need more sophisticated tracking to implement properly
    stats_.new_tracks_this_frame = 0;
    stats_.lost_tracks_this_frame = 0;
}

bool FusionModule::validateDetection(const Detection& detection) {
    // Basic validation checks
    
    // Check confidence threshold
    if (detection.confidence < 0.3) {
        return false;
    }
    
    // Check 2D bounding box validity
    if (detection.bbox_2d.area() > 0) {
        if (detection.bbox_2d.width < 10 || detection.bbox_2d.height < 10) {
            return false;
        }
        if (detection.bbox_2d.width > 1000 || detection.bbox_2d.height > 1000) {
            return false;
        }
    }
    
    // Check 3D position validity
    if (detection.center_3d.norm() > 0) {
        // Check if object is within reasonable range
        double distance = detection.center_3d.norm();
        if (distance > 100.0 || distance < 0.5) {
            return false;
        }
        
        // Check if object is above ground level (basic sanity check)
        if (detection.center_3d.z() < -3.0 || detection.center_3d.z() > 5.0) {
            return false;
        }
    }
    
    // Check 3D size validity
    if (detection.size_3d.norm() > 0) {
        if (detection.size_3d.x() > 10.0 || detection.size_3d.y() > 10.0 || 
            detection.size_3d.z() > 5.0) {
            return false;
        }
        if (detection.size_3d.x() < 0.1 || detection.size_3d.y() < 0.1 || 
            detection.size_3d.z() < 0.1) {
            return false;
        }
    }
    
    return true;
}

// AdvancedFusion Implementation
std::vector<TrackedObject*> AdvancedFusion::fuseTrackToTrack(
    const std::vector<TrackedObject*>& tracks_sensor1,
    const std::vector<TrackedObject*>& tracks_sensor2,
    double association_threshold) {
    
    std::vector<TrackedObject*> fused_tracks;
    std::vector<bool> sensor2_used(tracks_sensor2.size(), false);
    
    // Associate tracks from both sensors
    for (auto* track1 : tracks_sensor1) {
        double min_distance = std::numeric_limits<double>::max();
        int best_match = -1;
        
        for (size_t i = 0; i < tracks_sensor2.size(); ++i) {
            if (sensor2_used[i]) continue;
            
            double distance = calculateMahalanobisDistance(*track1, *tracks_sensor2[i]);
            if (distance < min_distance && distance < association_threshold) {
                min_distance = distance;
                best_match = i;
            }
        }
        
        if (best_match >= 0) {
            // Fuse the tracks (simplified - just use track1)
            fused_tracks.push_back(track1);
            sensor2_used[best_match] = true;
        } else {
            fused_tracks.push_back(track1);
        }
    }
    
    // Add unmatched tracks from sensor2
    for (size_t i = 0; i < tracks_sensor2.size(); ++i) {
        if (!sensor2_used[i]) {
            fused_tracks.push_back(tracks_sensor2[i]);
        }
    }
    
    return fused_tracks;
}

Eigen::MatrixXd AdvancedFusion::estimateAdaptiveNoise(
    const TrackedObject& track,
    const std::vector<Eigen::VectorXd>& recent_innovations) {
    
    if (recent_innovations.empty()) {
        return Eigen::MatrixXd::Identity(3, 3) * 0.1;
    }
    
    // Calculate sample covariance of innovations
    int n = recent_innovations.size();
    int dim = recent_innovations[0].size();
    
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
    for (const auto& innovation : recent_innovations) {
        mean += innovation;
    }
    mean /= n;
    
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(dim, dim);
    for (const auto& innovation : recent_innovations) {
        Eigen::VectorXd centered = innovation - mean;
        covariance += centered * centered.transpose();
    }
    covariance /= (n - 1);
    
    return covariance;
}

void AdvancedFusion::applyIMM(TrackedObject& track,
                             const std::vector<Eigen::MatrixXd>& motion_models,
                             double dt) {
    // Simplified IMM implementation
    // In a full implementation, this would maintain multiple filters
    // and compute model probabilities
    
    if (motion_models.empty() || !track.ekf->isInitialized()) {
        return;
    }
    
    // For now, just use the first motion model
    // A full IMM would run multiple filters and blend results
    track.ekf->predict(dt);
}

double AdvancedFusion::calculateMahalanobisDistance(const TrackedObject& track1,
                                                   const TrackedObject& track2) {
    
    if (!track1.ekf->isInitialized() || !track2.ekf->isInitialized()) {
        return std::numeric_limits<double>::max();
    }
    
    Eigen::VectorXd state1 = track1.ekf->getState();
    Eigen::VectorXd state2 = track2.ekf->getState();
    Eigen::MatrixXd cov1 = track1.ekf->getCovariance();
    Eigen::MatrixXd cov2 = track2.ekf->getCovariance();
    
    // Use position components only for distance calculation
    Eigen::Vector3d pos1 = state1.head<3>();
    Eigen::Vector3d pos2 = state2.head<3>();
    Eigen::Matrix3d pos_cov1 = cov1.block<3, 3>(0, 0);
    Eigen::Matrix3d pos_cov2 = cov2.block<3, 3>(0, 0);
    
    Eigen::Vector3d diff = pos1 - pos2;
    Eigen::Matrix3d combined_cov = pos_cov1 + pos_cov2;
    
    // Calculate Mahalanobis distance
    double distance = std::sqrt(diff.transpose() * combined_cov.inverse() * diff);
    
    return distance;
}

} // namespace tracking
