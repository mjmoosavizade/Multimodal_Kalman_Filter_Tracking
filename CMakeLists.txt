cmake_minimum_required(VERSION 3.16)
project(MultimodalKalmanFilterTracking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Find optional packages
find_package(PCL QUIET)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

# Conditional PCL support
if(PCL_FOUND)
    include_directories(${PCL_INCLUDE_DIRS})
    add_definitions(${PCL_DEFINITIONS})
    add_definitions(-DWITH_PCL)
    message(STATUS "PCL found - enabling full LiDAR support")
else()
    message(WARNING "PCL not found - LiDAR functionality will be limited")
endif()

# Source files (library)
set(LIB_SOURCES
    src/kalman_filter.cpp
    src/object_tracker.cpp
    src/data_loader.cpp
    src/perception_module.cpp
    src/fusion_module.cpp
    src/visualization.cpp
)

# Build library that can be linked by executables and tests
add_library(tracking_lib STATIC ${LIB_SOURCES})

# Ensure include dir is used by the library
target_include_directories(tracking_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link libraries to the tracking library
target_link_libraries(tracking_lib PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Conditional PCL linking for the library
if(PCL_FOUND)
    target_include_directories(tracking_lib PUBLIC ${PCL_INCLUDE_DIRS})
    target_compile_definitions(tracking_lib PUBLIC ${PCL_DEFINITIONS} -DWITH_PCL)
    target_link_libraries(tracking_lib PUBLIC ${PCL_LIBRARIES})
    message(STATUS "PCL found - enabling full LiDAR support")
else()
    message(WARNING "PCL not found - LiDAR functionality will be limited")
endif()

# Create executable (small main) that links the library
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE tracking_lib)

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Enable testing
enable_testing()
add_subdirectory(tests)
