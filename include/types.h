#pragma once

#ifdef WITH_PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
using PointCloudPtr = pcl::PointCloud<pcl::PointXYZI>::Ptr;
#else
#include <memory>
#include <vector>
struct SimplePoint { float x; float y; float z; float intensity; };
using PointCloudPtr = std::shared_ptr<std::vector<SimplePoint>>;
#endif
