#include "object_tracker.h"
#include <algorithm>
#include <iostream>
#include <limits>

namespace tracking {

MultiObjectTracker::MultiObjectTracker() 
    : next_track_id_(0), max_consecutive_misses_(5), association_threshold_(5.0) {
    initializeNoiseCovarianceMatrices();
}

void MultiObjectTracker::update(const std::vector<Detection>& detections,
                                const Eigen::Matrix3d& camera_matrix) {
    
    // Predict all existing tracks
    for (auto& [id, track] : tracks_) {
        if (track->ekf->isInitialized()) {
            track->ekf->predict(0.1); // Assume 10Hz update rate
        }
    }
    
    // Associate detections with existing tracks
    auto associations = associateDetections(detections);
    
    // Update tracks with associated detections
    for (const auto& [track_id, detection_idx] : associations) {
        if (track_id >= 0 && detection_idx >= 0) {
            auto& track = tracks_[track_id];
            const auto& detection = detections[detection_idx];
            
            // Update with camera measurement if available
            if (detection.bbox_2d.area() > 0) {
                Eigen::VectorXd camera_measurement(4);
                camera_measurement << detection.bbox_2d.x + detection.bbox_2d.width / 2,
                                     detection.bbox_2d.y + detection.bbox_2d.height / 2,
                                     detection.bbox_2d.width,
                                     detection.bbox_2d.height;
                
                track->ekf->updateCamera(camera_measurement, camera_matrix, R_camera_);
            }
            
            // Update with LiDAR measurement if available
            if (detection.center_3d.norm() > 0) {
                Eigen::VectorXd lidar_measurement(3);
                lidar_measurement << detection.center_3d.x(),
                                    detection.center_3d.y(),
                                    detection.center_3d.z();
                
                track->ekf->updateLiDAR(lidar_measurement, R_lidar_);
            }
            
            track->last_update_time = detection.timestamp;
            track->consecutive_misses = 0;
            track->confidence = std::min(1.0, track->confidence + 0.1);
        }
    }
    
    // Create new tracks for unassociated detections
    for (const auto& [track_id, detection_idx] : associations) {
        if (track_id == -1 && detection_idx >= 0) {
            createNewTrack(detections[detection_idx]);
        }
    }
    
    // Update miss counts for unassociated tracks
    for (auto& [id, track] : tracks_) {
        bool was_updated = false;
        for (const auto& [track_id, detection_idx] : associations) {
            if (track_id == id) {
                was_updated = true;
                break;
            }
        }
        
        if (!was_updated) {
            track->consecutive_misses++;
            track->confidence = std::max(0.0, track->confidence - 0.2);
        }
    }
    
    // Remove old tracks
    removeOldTracks();
}

std::vector<TrackedObject*> MultiObjectTracker::getActiveTracks() {
    std::vector<TrackedObject*> active_tracks;
    for (auto& [id, track] : tracks_) {
        if (track->consecutive_misses < max_consecutive_misses_ && track->confidence > 0.3) {
            active_tracks.push_back(track.get());
        }
    }
    return active_tracks;
}

TrackedObject* MultiObjectTracker::getTrack(int track_id) {
    auto it = tracks_.find(track_id);
    return (it != tracks_.end()) ? it->second.get() : nullptr;
}

std::vector<std::pair<int, int>> MultiObjectTracker::associateDetections(
    const std::vector<Detection>& detections) {
    
    std::vector<std::pair<int, int>> associations;
    std::vector<bool> detection_used(detections.size(), false);
    std::vector<bool> track_used(tracks_.size(), false);
    
    // Calculate distance matrix
    std::vector<std::vector<double>> distance_matrix;
    std::vector<int> track_ids;
    
    for (const auto& [id, track] : tracks_) {
        track_ids.push_back(id);
        std::vector<double> distances;
        
        for (const auto& detection : detections) {
            double distance = calculateDistance(*track, detection, Eigen::Matrix3d::Identity());
            distances.push_back(distance);
        }
        distance_matrix.push_back(distances);
    }
    
    // Simple greedy association (could be improved with Hungarian algorithm)
    for (size_t t = 0; t < track_ids.size(); ++t) {
        if (track_used[t]) continue;
        
        double min_distance = std::numeric_limits<double>::max();
        int best_detection = -1;
        
        for (size_t d = 0; d < detections.size(); ++d) {
            if (detection_used[d]) continue;
            
            if (distance_matrix[t][d] < min_distance && 
                distance_matrix[t][d] < association_threshold_) {
                min_distance = distance_matrix[t][d];
                best_detection = d;
            }
        }
        
        if (best_detection >= 0) {
            associations.push_back({track_ids[t], best_detection});
            track_used[t] = true;
            detection_used[best_detection] = true;
        }
    }
    
    // Add unassociated detections as new tracks
    for (size_t d = 0; d < detections.size(); ++d) {
        if (!detection_used[d]) {
            associations.push_back({-1, static_cast<int>(d)});
        }
    }
    
    return associations;
}

double MultiObjectTracker::calculateDistance(const TrackedObject& track,
                                           const Detection& detection,
                                           const Eigen::Matrix3d& camera_matrix) {
    
    if (!track.ekf->isInitialized()) {
        return std::numeric_limits<double>::max();
    }
    
    Eigen::VectorXd state = track.ekf->getState();
    
    // Calculate distance based on 3D position if available
    if (detection.center_3d.norm() > 0) {
        Eigen::Vector3d predicted_pos = state.head<3>();
        Eigen::Vector3d detection_pos = detection.center_3d;
        return (predicted_pos - detection_pos).norm();
    }
    
    // Otherwise use 2D distance
    if (detection.bbox_2d.area() > 0) {
        // Project predicted 3D position to 2D
        double px = state(0), py = state(1), pz = state(2);
        if (std::abs(pz) > 1e-6) {
            double u_pred = camera_matrix(0, 0) * px / pz + camera_matrix(0, 2);
            double v_pred = camera_matrix(1, 1) * py / pz + camera_matrix(1, 2);
            
            double u_det = detection.bbox_2d.x + detection.bbox_2d.width / 2;
            double v_det = detection.bbox_2d.y + detection.bbox_2d.height / 2;
            
            return std::sqrt((u_pred - u_det) * (u_pred - u_det) + 
                           (v_pred - v_det) * (v_pred - v_det));
        }
    }
    
    return std::numeric_limits<double>::max();
}

int MultiObjectTracker::createNewTrack(const Detection& detection) {
    int track_id = next_track_id_++;
    
    auto new_track = std::make_unique<TrackedObject>(track_id, detection.class_id);
    
    // Initialize state from detection
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(9);
    
    if (detection.center_3d.norm() > 0) {
        // Use 3D position from LiDAR
        initial_state.head<3>() = detection.center_3d;
    } else {
        // Estimate 3D position from 2D detection (simplified)
        initial_state(0) = detection.bbox_2d.x + detection.bbox_2d.width / 2;
        initial_state(1) = detection.bbox_2d.y + detection.bbox_2d.height / 2;
        initial_state(2) = 10.0; // Assume 10m distance
    }
    
    // Initialize velocity and acceleration to zero
    initial_state.segment<6>(3).setZero();
    
    new_track->ekf->initialize(initial_state);
    new_track->last_update_time = detection.timestamp;
    
    tracks_[track_id] = std::move(new_track);
    
    return track_id;
}

void MultiObjectTracker::removeOldTracks() {
    auto it = tracks_.begin();
    while (it != tracks_.end()) {
        if (it->second->consecutive_misses >= max_consecutive_misses_ ||
            it->second->confidence < 0.1) {
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }
}

void MultiObjectTracker::initializeNoiseCovarianceMatrices() {
    // Camera measurement noise (u, v, width, height)
    R_camera_ = Eigen::MatrixXd::Identity(4, 4);
    R_camera_(0, 0) = 5.0;  // u noise variance (pixels)
    R_camera_(1, 1) = 5.0;  // v noise variance (pixels)
    R_camera_(2, 2) = 10.0; // width noise variance (pixels)
    R_camera_(3, 3) = 10.0; // height noise variance (pixels)
    
    // LiDAR measurement noise (x, y, z)
    R_lidar_ = Eigen::MatrixXd::Identity(3, 3);
    R_lidar_(0, 0) = 0.1;  // x noise variance (meters)
    R_lidar_(1, 1) = 0.1;  // y noise variance (meters)
    R_lidar_(2, 2) = 0.1;  // z noise variance (meters)
}

} // namespace tracking
