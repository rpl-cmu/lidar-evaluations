/** @brief This module provides all functionality to extract LOAM features.
 * LOAM utilizes two types of features for data association and scan
 * registration 1) Planar Features 2) Edge Features Features are identified
 * based on "curvature" [1] Eq.(1) accounting for unreliable edge cases like
 * planes parallel to the LiDAR beam and points bordering potentially occluded
 * regions  [1] Fig. 4.
 *
 * Features are found independently in each scan line from the LiDAR. Thus LOAM
 * is reliant of having a structured pointcloud. Specifically, the feature
 * extraction module requires pointclouds to be in row major order. This is
 * intended to help with caching as each scan-line is searched over and thus
 * efficiency in the long run.
 *
 * [1] Ji Zhang and Sanjiv Singh, "LOAM: Lidar Odometry and Mapping in
 * Real-time," in Proceedings of Robotics: Science and Systems, 2014.
 *
 * @author Dan McGann
 * @date Mar 2024
 */
#pragma once
#include <memory>
#include <vector>

#include "../helpers.h"
#include "../types.h"
#include "mask.h"

/**
 * ######## ##    ## ########  ########  ######
 *    ##     ##  ##  ##     ## ##       ##    ##
 *    ##      ####   ##     ## ##       ##
 *    ##       ##    ########  ######    ######
 *    ##       ##    ##        ##             ##
 *    ##       ##    ##        ##       ##    ##
 *    ##       ##    ##        ########  ######
 */

/// @brief Structure for storing edge and planar feature points from a scan
/// together
/// @tparam PointType Template for point type see README.md
template <typename PointType, template <typename> class Alloc = std::allocator>
struct LoamFeatures {
  /// @brief A pointcloud of edge feature points
  std::vector<PointType, Alloc<PointType>> edge_points;
  /// @brief A pointcloud of planar feature points
  std::vector<PointType, Alloc<PointType>> planar_points;
};

/// @brief Structure for storing curvature information for points
struct PointCurvature {
  /// @brief The index of the point
  size_t index;
  /// @brief The curvature of the point
  double curvature;
  /// @brief Explicit parameterized constructor
  PointCurvature(size_t index, double curvature)
      : index(index), curvature(curvature) {}
  /// @brief Default constructor
  PointCurvature() = default;
};

/// @brief Comparator for PointCurvature used in std algorithms (like sort)
inline bool curvatureComparator(const PointCurvature &lhs,
                                const PointCurvature &rhs) {
  return lhs.curvature < rhs.curvature;
}

/**
 * #### ##    ## ######## ######## ########  ########    ###     ###### ########
 *  ##  ###   ##    ##    ##       ##     ## ##         ## ##   ##    ## ##
 *  ##  ####  ##    ##    ##       ##     ## ##        ##   ##  ##       ##
 *  ##  ## ## ##    ##    ######   ########  ######   ##     ## ##       ######
 *  ##  ##  ####    ##    ##       ##   ##   ##       ######### ##       ##
 *  ##  ##   ###    ##    ##       ##    ##  ##       ##     ## ##    ## ##
 * #### ##    ##    ##    ######## ##     ## ##       ##     ##  ###### ########
 */

/** @brief Extracts and returns LOAM features from a LiDAR scan. Main entry
 * point for using this module.
 * @param input_scan: The LiDAR Scan, organized in row-major order
 * @param lidar_params: The parameters for the lidar that observed input_scan
 * @tparam PointType Template for point type see README.md
 */
template <template <typename> class Accessor = FieldAccessor,
          typename PointType, template <typename> class Alloc>
LoamFeatures<PointType, Alloc> extractFeatures(
    const std::vector<PointType, Alloc<PointType>> &input_scan,
    const LidarParams &lidar_params,
    const FeatureExtractionParams &params = FeatureExtractionParams());

/** @brief Computes the curvature [1] Eq. (1) of each point in the given LiDAR
 * scan
 * @param input_scan: The LiDAR scan, organized in row-major order
 * @param lidar_params: The parameters for the lidar that observed input_scan
 * @tparam PointType Template for point type see README.md
 * @WARN To match the published LOAM implementation we do not normalize
 * curvature
 */
template <template <typename> class Accessor = FieldAccessor,
          typename PointType, template <typename> class Alloc>
std::vector<PointCurvature> computeCurvature(
    const std::vector<PointType, Alloc<PointType>> &input_scan,
    const LidarParams &lidar_params,
    const FeatureExtractionParams &params = FeatureExtractionParams());

/**
 * #### ##    ## ######## ######## ########  ##    ##    ###    ##
 *  ##  ###   ##    ##    ##       ##     ## ###   ##   ## ##   ##
 *  ##  ####  ##    ##    ##       ##     ## ####  ##  ##   ##  ##
 *  ##  ## ## ##    ##    ######   ########  ## ## ## ##     ## ##
 *  ##  ##  ####    ##    ##       ##   ##   ##  #### ######### ##
 *  ##  ##   ###    ##    ##       ##    ##  ##   ### ##     ## ##
 * #### ##    ##    ##    ######## ##     ## ##    ## ##     ## ########
 */

namespace features_internal {

/** @brief Converts features of PointType to the same features as
 * Eigen::Vector3d type This is necessary during registration to avoid repeated
 * conversions of points
 * @param in_features: The features to convert
 * @returns in_features with all points converted to eigen vectors
 */
template <template <typename> class Accessor = FieldAccessor,
          typename PointType, template <typename> class Alloc>
LoamFeatures<Eigen::Vector3d>
featuresToEigen(const LoamFeatures<PointType, Alloc> &in_features) {
  LoamFeatures<Eigen::Vector3d> result;
  for (const PointType pt : in_features.edge_points) {
    result.edge_points.push_back(pointToEigen<Accessor>(pt));
  }
  for (const PointType pt : in_features.planar_points) {
    result.planar_points.push_back(pointToEigen<Accessor>(pt));
  }
  return result;
}

/** @brief Extracts edge features and updates the mask for a scanline sector
 * defined by [sector_start, sector_end] This is used internally in feature
 * extraction, requires that curvature is sorted in range [sector_start,
 * sector_end] WARN: Mutates out_features adding the newly detected features
 * WARN: Mutates valid_mask marking neighbors of found features invalid
 * All param names match the local variable names in extractFeatures
 */
template <typename PointType, template <typename> class Alloc>
void extractSectorEdgeFeatures(
    const size_t &sector_start_point, const size_t &sector_end_point,
    const std::vector<PointType, Alloc<PointType>> &input_scan,
    const std::vector<PointCurvature> &curvature,
    const FeatureExtractionParams &params,
    LoamFeatures<PointType, Alloc> &out_features,
    std::vector<bool> &valid_mask);

/** @brief Extracts planar features and updates the mask for a scanline sector
 * defined by [sector_start, sector_end] This is used internally in feature
 * extraction, requires that curvature is sorted in range [sector_start,
 * sector_end] WARN: Mutates out_features adding the newly detected features
 * WARN: Mutates valid_mask marking neighbors of found features invalid
 *
 * All param names match the local variable names in extractFeatures
 */
template <typename PointType, template <typename> class Alloc>
void extractSectorPlanarFeatures(
    const size_t &sector_start_point, const size_t &sector_end_point,
    const std::vector<PointType, Alloc<PointType>> &input_scan,
    const std::vector<PointCurvature> &curvature,
    const FeatureExtractionParams &params,
    LoamFeatures<PointType, Alloc> &out_features,
    std::vector<bool> &valid_mask);

} // namespace features_internal

// Include the actual implementation of this module
#include "features-inl.h"