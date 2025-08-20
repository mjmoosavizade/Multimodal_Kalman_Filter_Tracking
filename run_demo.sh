#!/bin/bash

echo "🎬 Building and Running Kalman Filter Tracking Demo"
echo "=================================================="

# Build the project
echo "📦 Building project..."
if [ ! -d "build" ]; then
    mkdir build
fi

cd build
cmake .. && make demo_visualization

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "🚀 Starting visualization demo..."
    echo "Controls:"
    echo "  - Press 'q' to quit"
    echo "  - Press 'p' to pause/unpause"
    echo "  - Press 's' to save current frame"
    echo "  - Press 'r' to start/stop video recording"
    echo ""
    echo "🎯 Features demonstrated:"
    echo "  - Extended Kalman Filter tracking (9D state)"
    echo "  - Multi-object tracking with data association"
    echo "  - Real-time trajectory prediction"
    echo "  - Confidence scoring and classification"
    echo "  - Professional visualization with metrics"
    echo ""
    
    ./demo_visualization
    
    echo ""
    echo "✨ Demo completed!"
    if [ -f "tracking_demo.mp4" ]; then
        echo "📹 Video recording saved as: tracking_demo.mp4"
    fi
    
    if ls tracking_demo_*.png 1> /dev/null 2>&1; then
        echo "📸 Screenshots saved as: tracking_demo_*.png"
    fi
else
    echo "❌ Build failed!"
    exit 1
fi
