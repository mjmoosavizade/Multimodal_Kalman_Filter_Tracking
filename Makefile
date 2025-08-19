# Makefile for Multimodal Kalman Filter Tracking

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native
DEBUG_FLAGS = -g -DDEBUG

# Include directories
INCLUDES = -Iinclude \
           $(shell pkg-config --cflags opencv4) \
           $(shell pkg-config --cflags eigen3) \
           -I/usr/include/pcl-1.12 \
           -I/usr/include/vtk-9.1

# Library flags
LIBS = $(shell pkg-config --libs opencv4) \
       -lpcl_common -lpcl_io -lpcl_filters -lpcl_segmentation \
       -lpcl_visualization -lpcl_search -lpcl_kdtree \
       -lvtkCommonCore -lvtkRenderingCore -lvtkRenderingLOD \
       -lvtkRenderingOpenGL2 -lvtkInteractionStyle

# Source files
SRCDIR = src
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

# Target executable
TARGET = MultimodalKalmanFilterTracking

# Example target
EXAMPLE_TARGET = simple_demo
EXAMPLE_SOURCES = examples/simple_demo.cpp $(filter-out src/main.cpp, $(SOURCES))
EXAMPLE_OBJECTS = $(EXAMPLE_SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Main executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Example executable
$(EXAMPLE_TARGET): $(EXAMPLE_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Debug build
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: $(TARGET)

# Clean build files
clean:
	rm -f $(OBJECTS) $(EXAMPLE_OBJECTS) $(TARGET) $(EXAMPLE_TARGET)
	rm -rf build/

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt update
	sudo apt install -y \
		libopencv-dev \
		libpcl-dev \
		libeigen3-dev \
		cmake \
		build-essential \
		pkg-config

# Create build directory and use CMake
cmake-build:
	mkdir -p build
	cd build && cmake .. && make -j$$(nproc)

# Run with sample data (requires KITTI dataset)
run-sample:
	@echo "Please provide KITTI dataset path:"
	@echo "make run DATASET_PATH=/path/to/kitti/sequence"

run:
	@if [ -z "$(DATASET_PATH)" ]; then \
		echo "Error: DATASET_PATH not specified"; \
		echo "Usage: make run DATASET_PATH=/path/to/kitti/sequence"; \
		exit 1; \
	fi
	./$(TARGET) $(DATASET_PATH)

# Run simple demo
demo: $(EXAMPLE_TARGET)
	./$(EXAMPLE_TARGET)

# Format code (requires clang-format)
format:
	find src include examples tests -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# Static analysis (requires cppcheck)
analyze:
	cppcheck --enable=all --std=c++17 src/ include/

# Generate documentation (requires doxygen)
docs:
	doxygen Doxyfile

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build main executable"
	@echo "  debug        - Build with debug flags"
	@echo "  demo         - Build and run simple demo"
	@echo "  clean        - Remove build files"
	@echo "  install-deps - Install system dependencies"
	@echo "  cmake-build  - Build using CMake"
	@echo "  run          - Run with KITTI dataset (specify DATASET_PATH)"
	@echo "  format       - Format code with clang-format"
	@echo "  analyze      - Run static analysis with cppcheck"
	@echo "  docs         - Generate documentation with doxygen"
	@echo "  help         - Show this help message"

.PHONY: all debug clean install-deps cmake-build run run-sample demo format analyze docs help
