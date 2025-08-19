#!/bin/bash

# Build script for Multimodal Kalman Filter Tracking

set -e  # Exit on any error

echo "=== Building Multimodal Kalman Filter Tracking ==="

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building project..."
make -j$(nproc)

echo "Build completed successfully!"
echo ""
echo "To run the program:"
echo "  ./MultimodalKalmanFilterTracking /path/to/kitti/dataset"
echo ""
echo "To run tests (if Google Test is available):"
echo "  cd tests && ./run_tests"
