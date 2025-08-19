# Multimodal Kalman Filter Tracking

A robust 3D object tracking system that combines AI-based perception with Extended Kalman Filters for vehicle tracking in autonomous driving scenarios. The system fuses camera and LiDAR sensor data using probabilistic filtering to achieve accurate and stable tracking.

## Overview

This project implements a comprehensive multimodal tracking system designed for autonomous vehicles. It leverages the strengths of both AI-based object detection and traditional probabilistic filtering to create a robust tracking solution.

### Key Features

- **Multimodal Sensor Fusion**: Combines camera images and LiDAR point clouds
- **Extended Kalman Filter**: 9-DOF state estimation (position, velocity, acceleration)
- **AI-Powered Detection**: Deep learning models for object detection
- **Real-time Visualization**: 2D and 3D visualization of tracking results
- **KITTI Dataset Support**: Compatible with KITTI tracking dataset
- **Performance Profiling**: Built-in performance monitoring and statistics

## Architecture

The system follows a three-stage pipeline:

### 1. Data Ingestion & Perception
- **Camera Processing**: AI-based object detection (YOLO/Faster R-CNN compatible)
- **LiDAR Processing**: 3D clustering and geometric classification
- **IMU/GPS Integration**: Ego-motion compensation

### 2. Fusion and Prediction
- **Extended Kalman Filter**: Maintains 9D state vector [px, py, pz, vx, vy, vz, ax, ay, az]
- **Multi-Object Tracking**: Handles multiple objects simultaneously
- **Data Association**: Robust assignment of detections to tracks

### 3. Output and Visualization
- **2D Visualization**: Annotated camera images with tracking results
- **3D Visualization**: Point cloud viewer with 3D bounding boxes
- **Performance Metrics**: Real-time tracking statistics

## Dependencies

### Required Libraries
- **OpenCV** (≥ 4.0): Computer vision and image processing
- **PCL** (≥ 1.8): Point cloud processing
- **Eigen3** (≥ 3.3): Linear algebra operations
- **CMake** (≥ 3.16): Build system

### Installation (Ubuntu/Debian)

```bash
# Install OpenCV
sudo apt update
sudo apt install libopencv-dev

# Install PCL
sudo apt install libpcl-dev

# Install Eigen3
sudo apt install libeigen3-dev

# Install CMake
sudo apt install cmake build-essential
```

### Installation (macOS with Homebrew)

```bash
brew install opencv pcl eigen cmake
```

## Building the Project

```bash
# Clone the repository
git clone <repository-url>
cd Multimodal_Kalman_Filter_Tracking

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j$(nproc)
```

## Usage

### Basic Usage with KITTI Dataset

```bash
# Run with KITTI dataset
./MultimodalKalmanFilterTracking /path/to/kitti/sequence_00

# Run with custom detection model
./MultimodalKalmanFilterTracking /path/to/kitti/sequence_00 /path/to/model.onnx
```

### KITTI Dataset Structure

Ensure your KITTI dataset follows this structure:
```
sequence_00/
├── image_00/
│   ├── data/
│   │   ├── 0000000000.png
│   │   └── ...
│   └── timestamps.txt
├── image_01/
├── velodyne_points/
│   ├── data/
│   │   ├── 0000000000.bin
│   │   └── ...
│   └── timestamps.txt
├── oxts/
│   ├── data/
│   │   ├── 0000000000.txt
│   │   └── ...
│   └── timestamps.txt
└── calib.txt
```

### Interactive Controls

- **'q' or ESC**: Quit the application
- **'p'**: Pause/resume processing
- **'s'**: Save current frame as image

## Configuration

### Kalman Filter Parameters

The Extended Kalman Filter can be tuned by modifying parameters in `kalman_filter.cpp`:

```cpp
// Process noise (acceleration uncertainty)
double sigma_a = 2.0; // m/s²

// Measurement noise covariances
R_camera_ = diag([5.0, 5.0, 10.0, 10.0]); // pixels
R_lidar_ = diag([0.1, 0.1, 0.1]);         // meters
```

### Tracking Parameters

Adjust tracking behavior in `fusion_module.cpp`:

```cpp
// Association threshold (distance metric)
double association_threshold = 5.0;

// Maximum consecutive misses before track deletion
int max_consecutive_misses = 5;

// Confidence thresholds
double min_detection_confidence = 0.3;
double min_track_confidence = 0.1;
```

## System Components

### Core Classes

1. **ExtendedKalmanFilter**: Implements EKF for state estimation
2. **MultiObjectTracker**: Manages multiple object tracks
3. **PerceptionModule**: Handles multimodal object detection
4. **FusionModule**: Coordinates sensor fusion and tracking
5. **Visualizer**: Provides 2D and 3D visualization

### Data Structures

```cpp
// Detection structure
struct Detection {
    int class_id;           // Object class (0: car, 1: pedestrian, 2: cyclist)
    double confidence;      // Detection confidence [0, 1]
    cv::Rect2d bbox_2d;     // 2D bounding box
    Eigen::Vector3d center_3d;  // 3D center position
    Eigen::Vector3d size_3d;     // 3D size (width, height, depth)
    double timestamp;       // Detection timestamp
};

// Tracked object structure
struct TrackedObject {
    int id;                 // Unique track ID
    int class_id;          // Object class
    ExtendedKalmanFilter* ekf;  // Kalman filter
    double last_update_time;    // Last update timestamp
    int consecutive_misses;     // Consecutive missed detections
    double confidence;          // Track confidence
};
```

## Performance Optimization

### Computational Efficiency
- Voxel grid filtering for point cloud downsampling
- Efficient data association algorithms
- Optimized matrix operations using Eigen

### Memory Management
- Smart pointers for automatic memory management
- Efficient data structures for large point clouds
- Configurable buffer sizes

## Extending the System

### Adding New Sensors
1. Implement sensor-specific detection in `PerceptionModule`
2. Add measurement model to `ExtendedKalmanFilter`
3. Update fusion logic in `FusionModule`

### Custom Detection Models
1. Modify `CameraDetector::initialize()` for new model formats
2. Update `postprocessOutputs()` for different output formats
3. Adjust class mappings in `getClassName()`

### Advanced Filtering
- Implement Unscented Kalman Filter (UKF)
- Add Interacting Multiple Model (IMM) filters
- Implement particle filters for non-linear scenarios

## Troubleshooting

### Common Issues

1. **Build Errors**
   - Ensure all dependencies are installed
   - Check CMake version compatibility
   - Verify library paths in CMakeLists.txt

2. **Runtime Errors**
   - Verify KITTI dataset structure
   - Check file permissions
   - Ensure sufficient memory for point cloud processing

3. **Performance Issues**
   - Reduce point cloud density with voxel filtering
   - Adjust visualization update rates
   - Use optimized compiler flags (-O3)

### Debug Mode

Build with debug information:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- KITTI Dataset: [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)
- Point Cloud Library (PCL): [https://pointclouds.org/](https://pointclouds.org/)
- OpenCV: [https://opencv.org/](https://opencv.org/)
- Eigen: [https://eigen.tuxfamily.org/](https://eigen.tuxfamily.org/)

## References

1. Kalman, R. E. (1960). "A new approach to linear filtering and prediction problems"
2. Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). "Estimation with applications to tracking and navigation"
3. Geiger, A., Lenz, P., & Urtasun, R. (2012). "Are we ready for autonomous driving? The KITTI vision benchmark suite"

## Contact

For questions and support, please open an issue on the GitHub repository.
