#pragma once
#pragma once

#include "object_tracker.h"
#include "types.h"
#include "fusion_module.h"
#include <opencv2/opencv.hpp>
#ifdef WITH_PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#endif
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace tracking {

/**
 * @brief Visualization module for tracking results
 */
class Visualizer {
public:
    Visualizer();
    ~Visualizer() = default;

    /**
     * @brief Initialize visualizer
     * @param window_name Window name for display
     * @param enable_3d_viewer Enable 3D point cloud viewer
     * @return True if successful
     */
    bool initialize(const std::string& window_name = "Multimodal Tracking",
                   bool enable_3d_viewer = true);

    /**
     * @brief Visualize tracking results on image
     * @param image Input image
     * @param tracks Active tracks
     * @param detections Current detections
     * @param camera_matrix Camera intrinsic matrix
     * @param show_predictions Show predicted positions
     * @return Annotated image
     */
    cv::Mat visualizeImage(const cv::Mat& image,
                          const std::vector<tracking::TrackedObject*>& tracks,
                          const std::vector<Detection>& detections,
                          const Eigen::Matrix3d& camera_matrix,
                          bool show_predictions = true);

    /**
     * @brief Visualize tracking results in 3D point cloud
     * @param cloud Input point cloud
     * @param tracks Active tracks
     * @param detections Current detections
     * @param ego_position Current ego vehicle position
     */
    void visualize3D(const PointCloudPtr& cloud,
                    const std::vector<tracking::TrackedObject*>& tracks,
                    const std::vector<Detection>& detections,
                    const Eigen::Vector3d& ego_position = Eigen::Vector3d::Zero());

    /**
     * @brief Update 3D visualization
     * @param spin_once If true, call spinOnce(), otherwise spin()
     */
    void update3DVisualization(bool spin_once = true);

    /**
     * @brief Save visualization results
     * @param image_path Path to save annotated image
     * @param cloud_path Path to save point cloud (optional)
     * @return True if successful
     */
    bool saveResults(const std::string& image_path,
                    const std::string& cloud_path = "");

    /**
     * @brief Create tracking summary video
     * @param output_path Output video path
     * @param fps Video frame rate
     * @return True if successful
     */
    bool createSummaryVideo(const std::string& output_path, double fps = 30.0);

    /**
     * @brief Display tracking statistics
     * @param image Image to overlay statistics on
     * @param stats Tracking statistics
     * @return Image with statistics overlay
     */
    cv::Mat displayStatistics(const cv::Mat& image,
                             const FusionModule::TrackingStats& stats);

private:
    std::string window_name_;
    bool enable_3d_viewer_;
#ifdef WITH_PCL
    pcl::visualization::PCLVisualizer::Ptr viewer_3d_;
#else
    // viewer_3d_ is only available when PCL is enabled
    void* viewer_3d_ = nullptr;
#endif
    cv::VideoWriter video_writer_;
    
    // Color palette for different tracks
    std::vector<cv::Scalar> track_colors_;
    
    // Visualization parameters
    struct VisualizationParams {
        int bbox_thickness = 2;
        int text_thickness = 1;
        double text_scale = 0.6;
        int trajectory_length = 10;
        bool show_velocity_vectors = true;
        bool show_uncertainty_ellipse = true;
        double prediction_time = 1.0; // seconds
    } params_;

    /**
     * @brief Initialize color palette for tracks
     */
    void initializeColorPalette();

    /**
     * @brief Get color for specific track ID
     * @param track_id Track identifier
     * @return BGR color
     */
    cv::Scalar getTrackColor(int track_id);

    /**
     * @brief Draw 2D bounding box on image
     * @param image Image to draw on
     * @param bbox Bounding box
     * @param color Box color
     * @param thickness Line thickness
     * @param label Text label
     */
    void drawBoundingBox2D(cv::Mat& image,
                          const cv::Rect2d& bbox,
                          const cv::Scalar& color,
                          int thickness = 2,
                          const std::string& label = "");

    /**
     * @brief Draw 3D bounding box projected to image
     * @param image Image to draw on
     * @param center_3d 3D center position
     * @param size_3d 3D size
     * @param camera_matrix Camera intrinsic matrix
     * @param color Box color
     * @param thickness Line thickness
     */
    void drawBoundingBox3D(cv::Mat& image,
                          const Eigen::Vector3d& center_3d,
                          const Eigen::Vector3d& size_3d,
                          const Eigen::Matrix3d& camera_matrix,
                          const cv::Scalar& color,
                          int thickness = 2);

    /**
     * @brief Draw velocity vector
     * @param image Image to draw on
     * @param position Current position
     * @param velocity Velocity vector
     * @param camera_matrix Camera intrinsic matrix
     * @param color Vector color
     * @param scale Scaling factor
     */
    void drawVelocityVector(cv::Mat& image,
                           const Eigen::Vector3d& position,
                           const Eigen::Vector3d& velocity,
                           const Eigen::Matrix3d& camera_matrix,
                           const cv::Scalar& color,
                           double scale = 1.0);

    /**
     * @brief Draw uncertainty ellipse
     * @param image Image to draw on
     * @param center Center position
     * @param covariance Covariance matrix (2x2)
     * @param color Ellipse color
     * @param confidence Confidence level (e.g., 0.95)
     */
    void drawUncertaintyEllipse(cv::Mat& image,
                               const cv::Point2f& center,
                               const Eigen::Matrix2d& covariance,
                               const cv::Scalar& color,
                               double confidence = 0.95);

    /**
     * @brief Project 3D point to 2D image coordinates
     * @param point_3d 3D point
     * @param camera_matrix Camera intrinsic matrix
     * @return 2D image point
     */
    cv::Point2f project3DTo2D(const Eigen::Vector3d& point_3d,
                              const Eigen::Matrix3d& camera_matrix);

    /**
     * @brief Add 3D bounding box to point cloud viewer
     * @param center_3d 3D center position
     * @param size_3d 3D size
     * @param id Unique identifier for the box
     * @param color RGB color (0-1 range)
     */
    void add3DBoundingBox(const Eigen::Vector3d& center_3d,
                         const Eigen::Vector3d& size_3d,
                         const std::string& id,
                         const std::array<double, 3>& color);

    /**
     * @brief Convert BGR color to RGB (0-1 range)
     * @param bgr_color BGR color (0-255 range)
     * @return RGB color (0-1 range)
     */
    std::array<double, 3> bgrToRgb(const cv::Scalar& bgr_color);

    /**
     * @brief Get class name from class ID
     * @param class_id Class identifier
     * @return Class name string
     */
    std::string getClassName(int class_id);

    /**
     * @brief Add legend to visualization image
     * @param image Image to add legend to
     */
    void addLegend(cv::Mat& image);
};

/**
 * @brief Performance profiler for tracking system
 */
class PerformanceProfiler {
public:
    PerformanceProfiler();
    ~PerformanceProfiler() = default;

    /**
     * @brief Start timing a specific operation
     * @param operation_name Name of the operation
     */
    void startTimer(const std::string& operation_name);

    /**
     * @brief Stop timing an operation
     * @param operation_name Name of the operation
     */
    void stopTimer(const std::string& operation_name);

    /**
     * @brief Get average execution time for an operation
     * @param operation_name Name of the operation
     * @return Average time in milliseconds
     */
    double getAverageTime(const std::string& operation_name) const;

    /**
     * @brief Print performance summary
     */
    void printSummary() const;

    /**
     * @brief Reset all timers
     */
    void reset();

private:
    struct TimingData {
        std::chrono::high_resolution_clock::time_point start_time;
        std::vector<double> execution_times;
        bool is_running = false;
    };

    std::unordered_map<std::string, TimingData> timers_;
};

} // namespace tracking
