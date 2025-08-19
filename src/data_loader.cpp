#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#ifdef WITH_PCL
#include <pcl/io/pcd_io.h>
#endif

namespace tracking {

KITTIDataLoader::KITTIDataLoader(const std::string& dataset_path) 
    : dataset_path_(dataset_path), num_frames_(0) {
}

bool KITTIDataLoader::initialize() {
    // Load calibration data
    if (!loadCalibrationData()) {
        std::cerr << "Failed to load calibration data" << std::endl;
        return false;
    }
    
    // Load timestamps
    if (!loadTimestamps()) {
        std::cerr << "Failed to load timestamps" << std::endl;
        return false;
    }
    
    // Determine number of frames
    std::string image_dir = dataset_path_ + "/image_00/data";
    if (std::filesystem::exists(image_dir)) {
        num_frames_ = 0;
        for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
            if (entry.path().extension() == ".png") {
                num_frames_++;
            }
        }
    }
    
    std::cout << "Initialized KITTI data loader with " << num_frames_ << " frames" << std::endl;
    return num_frames_ > 0;
}

cv::Mat KITTIDataLoader::loadCameraImage(int camera_id, int frame_index) {
    if (!isValidFrame(frame_index) || camera_id < 0 || camera_id > 3) {
        return cv::Mat();
    }
    
    std::stringstream ss;
    ss << dataset_path_ << "/image_" << std::setfill('0') << std::setw(2) << camera_id 
       << "/data/" << std::setfill('0') << std::setw(10) << frame_index << ".png";
    
    std::string image_path = ss.str();
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
    }
    
    return image;
}

// Conditional implementation depending on PCL availability
#ifdef WITH_PCL
PointCloudPtr KITTIDataLoader::loadLiDARPointCloud(int frame_index) {
    // When PCL is available, return pcl::PointCloud pointer
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    if (!isValidFrame(frame_index)) {
        return cloud;
    }

    std::stringstream ss;
    ss << dataset_path_ << "/velodyne_points/data/"
       << std::setfill('0') << std::setw(10) << frame_index << ".bin";

    std::string velodyne_path = ss.str();
    std::ifstream file(velodyne_path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Failed to open LiDAR file: " << velodyne_path << std::endl;
        return cloud;
    }

    // Each point in KITTI velodyne binary has 4 floats: x, y, z, reflectance
    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t num_points = file_size / (4 * sizeof(float));

    cloud->resize(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        float x, y, z, intensity;
        file.read(reinterpret_cast<char*>(&x), sizeof(float));
        file.read(reinterpret_cast<char*>(&y), sizeof(float));
        file.read(reinterpret_cast<char*>(&z), sizeof(float));
        file.read(reinterpret_cast<char*>(&intensity), sizeof(float));
        pcl::PointXYZI pt;
        pt.x = x; pt.y = y; pt.z = z; pt.intensity = intensity;
        cloud->at(i) = pt;
    }

    return cloud;
}
#else
PointCloudPtr KITTIDataLoader::loadLiDARPointCloud(int frame_index) {
    // PCL not available: return an empty simple point cloud
    auto cloud = std::make_shared<std::vector<SimplePoint>>();
    (void)frame_index; // unused
    return cloud;
}
#endif

IMUData KITTIDataLoader::loadIMUData(int frame_index) {
    IMUData imu_data;
    
    if (!isValidFrame(frame_index)) {
        return imu_data;
    }
    
    std::stringstream ss;
    ss << dataset_path_ << "/oxts/data/" 
       << std::setfill('0') << std::setw(10) << frame_index << ".txt";
    
    std::string oxts_path = ss.str();
    std::ifstream file(oxts_path);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open OXTS file: " << oxts_path << std::endl;
        return imu_data;
    }
    
    std::string line;
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> values;
        std::string token;
        
        while (std::getline(iss, token, ' ')) {
            values.push_back(std::stod(token));
        }
        
        if (values.size() >= 30) {
            // OXTS format: lat, lon, alt, roll, pitch, yaw, vn, ve, vf, vl, vu, 
            // ax, ay, az, af, al, au, wx, wy, wz, wf, wl, wu, pos_accuracy, vel_accuracy, ...
            
            imu_data.position << values[0], values[1], values[2];  // lat, lon, alt
            imu_data.velocity << values[6], values[7], values[8];  // vn, ve, vf
            imu_data.linear_acceleration << values[11], values[12], values[13]; // ax, ay, az
            imu_data.angular_velocity << values[17], values[18], values[19];    // wx, wy, wz
            
            if (frame_index < static_cast<int>(oxts_timestamps_.size())) {
                imu_data.timestamp = oxts_timestamps_[frame_index];
            }
        }
    }
    
    file.close();
    return imu_data;
}

Eigen::Matrix3d KITTIDataLoader::getCameraMatrix(int camera_id) const {
    if (camera_id >= 0 && camera_id < static_cast<int>(camera_matrices_.size())) {
        return camera_matrices_[camera_id];
    }
    return Eigen::Matrix3d::Identity();
}

Eigen::Matrix4d KITTIDataLoader::getCameraToLiDARTransform(int camera_id) const {
    if (camera_id >= 0 && camera_id < static_cast<int>(camera_to_lidar_transforms_.size())) {
        return camera_to_lidar_transforms_[camera_id];
    }
    return Eigen::Matrix4d::Identity();
}

double KITTIDataLoader::getTimestamp(int frame_index) const {
    if (frame_index >= 0 && frame_index < static_cast<int>(timestamps_.size())) {
        return timestamps_[frame_index];
    }
    return 0.0;
}

bool KITTIDataLoader::isValidFrame(int frame_index) const {
    return frame_index >= 0 && frame_index < num_frames_;
}

bool KITTIDataLoader::loadCalibrationData() {
    std::string calib_path = dataset_path_ + "/calib.txt";
    
    if (!std::filesystem::exists(calib_path)) {
        // Try alternative path
        calib_path = dataset_path_ + "/calib_cam_to_cam.txt";
    }
    
    if (!std::filesystem::exists(calib_path)) {
        std::cerr << "Calibration file not found" << std::endl;
        return false;
    }
    
    return parseCalibrationFile(calib_path);
}

bool KITTIDataLoader::loadTimestamps() {
    // Load camera timestamps
    std::string timestamp_path = dataset_path_ + "/image_00/timestamps.txt";
    if (!parseTimestampFile(timestamp_path, timestamps_)) {
        std::cerr << "Failed to load camera timestamps" << std::endl;
        return false;
    }
    
    // Load velodyne timestamps
    timestamp_path = dataset_path_ + "/velodyne_points/timestamps.txt";
    if (!parseTimestampFile(timestamp_path, velodyne_timestamps_)) {
        std::cerr << "Failed to load velodyne timestamps" << std::endl;
        return false;
    }
    
    // Load OXTS timestamps
    timestamp_path = dataset_path_ + "/oxts/timestamps.txt";
    if (!parseTimestampFile(timestamp_path, oxts_timestamps_)) {
        std::cerr << "Failed to load OXTS timestamps" << std::endl;
        return false;
    }
    
    return true;
}

bool KITTIDataLoader::parseCalibrationFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    camera_matrices_.resize(4);
    camera_to_lidar_transforms_.resize(4);
    
    // Initialize with identity matrices
    for (int i = 0; i < 4; ++i) {
        camera_matrices_[i] = Eigen::Matrix3d::Identity();
        camera_to_lidar_transforms_[i] = Eigen::Matrix4d::Identity();
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        
        if (key.substr(0, 2) == "P_" || key.substr(0, 2) == "P0" || 
            key.substr(0, 2) == "P1" || key.substr(0, 2) == "P2" || key.substr(0, 2) == "P3") {
            
            int cam_id = 0;
            if (key.length() > 2) {
                cam_id = key[2] - '0';
            }
            
            if (cam_id >= 0 && cam_id < 4) {
                std::vector<double> values;
                double value;
                while (iss >> value) {
                    values.push_back(value);
                }
                
                if (values.size() >= 12) {
                    // Extract 3x3 camera matrix from 3x4 projection matrix
                    camera_matrices_[cam_id] << values[0], values[1], values[2],
                                               values[4], values[5], values[6],
                                               values[8], values[9], values[10];
                }
            }
        }
    }
    
    file.close();
    return true;
}

bool KITTIDataLoader::parseTimestampFile(const std::string& filepath, 
                                        std::vector<double>& timestamps) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    timestamps.clear();
    std::string line;
    while (std::getline(file, line)) {
        double timestamp = parseTimestamp(line);
        timestamps.push_back(timestamp);
    }
    
    file.close();
    return !timestamps.empty();
}

double KITTIDataLoader::parseTimestamp(const std::string& timestamp_str) {
    // KITTI timestamp format: YYYY-MM-DD HH:MM:SS.ssssss
    // Convert to seconds since epoch for simplicity
    
    std::tm tm = {};
    std::istringstream ss(timestamp_str);
    
    // Parse date and time
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    
    if (ss.fail()) {
        return 0.0;
    }
    
    // Convert to time_t
    std::time_t time = std::mktime(&tm);
    
    // Extract microseconds
    size_t dot_pos = timestamp_str.find('.');
    double microseconds = 0.0;
    if (dot_pos != std::string::npos && dot_pos + 1 < timestamp_str.length()) {
        std::string microsec_str = timestamp_str.substr(dot_pos + 1);
        microseconds = std::stod("0." + microsec_str);
    }
    
    return static_cast<double>(time) + microseconds;
}

} // namespace tracking
