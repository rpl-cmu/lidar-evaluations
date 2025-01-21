#pragma once

#include "../helpers.h"
#include "../types.h"
#include <vector>

/// @brief Structure for storing feature extraction parameters
struct FeatureExtractionParams {
  /// @brief The number of neighbor points (on either size) to use when
  /// computing curvature [1] Eq. (1) A reasonable number is between 3 and 6,
  /// less and curvature is too noisy, more and points cover too large an area
  size_t neighbor_points{5};
  /// @brief The number of sectors to break each scan line into when detecting
  /// feature points A reasonable number is between 4 and 8
  // If the number of points per line is not divisible by number_sectors,
  // remainder points are added to the last sector
  size_t number_sectors{6};
  /// @brief The maximum number of edge features to detect in each sector
  /// Reasonable numbers depends on compute power available for registration,
  /// with more points registration is expensive
  size_t max_edge_feats_per_sector{5};
  /// @brief The maximum number of planar features to detect in each sector
  /// Reasonable numbers depends on compute power available for registration,
  /// with more points registration is expensive
  size_t max_planar_feats_per_sector{5};
  /// @brief Threshold for edge feature curvature.
  /// The UNNORMALIZED curvature must be greater than this thresh to be
  /// considered an edge feature. WARN: This is an unintuitive param manual
  /// tuning and plotting results is recommended
  double edge_feat_threshold{100.0};
  /// @brief Threshold for planar feature curvature.
  /// The UNNORMALIZED curvature must be less than this thresh to be considered
  /// a planar feature. WARN: This is an unintuitive param manual tuning and
  /// plotting results is recommended
  double planar_feat_threshold{0.1};
  /// @brief This distance in point units (e.x. meters) between neighboring
  /// points to be considered for occlusion Reasonable values are on the order
  /// of 1m for most robotics applications
  double occlusion_thresh{0.25};
  /// @brief The range difference (as proportion of range) between consecutive
  /// points to be considered too parallel See computeValidPoints::Check 4 for
  /// details WARN: This is an unintuitive param manual tuning and plotting
  /// results is recommended
  double parallel_thresh{0.002};
};

namespace features_internal {
/** @brief Marks edge points as invalid (see computeValidPoints) returns true if
 * the point is marked WARN: Potentially mutates mask if the point is invalid
 * All param names match the local variable names in computeValidPoints
 * @returns true if the point was marked invalid in the mask
 */
bool markEdgesInvalid(const size_t &idx, const size_t &line_pt_idx,
                      const LidarParams &lidar_params,
                      const FeatureExtractionParams &params,
                      std::vector<bool> &mask);

/** @brief Marks out of range points as invalid (see computeValidPoints) returns
 * true if the point is marked WARN: Potentially mutates mask if the point is
 * invalid All param names match the local variable names in computeValidPoints
 * @returns true if the point was marked invalid in the mask
 */
bool markOutOfRangeInvalid(const size_t &idx, const double &point_range,
                           const LidarParams &lidar_params,
                           const FeatureExtractionParams &params,
                           std::vector<bool> &mask);

/** @brief Marks occluded points as invalid (see computeValidPoints) returns
 * true if the point is marked WARN: Potentially mutates mask if the point is
 * invalid All param names match the local variable names in computeValidPoints
 * @returns true if the point was marked invalid in the mask
 */
bool markOccludedInvalid(const size_t &idx, const double &point_range,
                         const double &next_point_range,
                         const FeatureExtractionParams &params,
                         std::vector<bool> &mask);

/** @brief Marks occluded points as invalid (see computeValidPoints) returns
 * true if the point is marked WARN: Potentially mutates mask if the point is
 * invalid All param names match the local variable names in computeValidPoints
 * @returns true if the point was marked invalid in the mask
 */
bool markParallelInvalid(const size_t &idx, const double &prev_point_range,
                         const double &point_range,
                         const double &next_point_range,
                         const FeatureExtractionParams &params,
                         std::vector<bool> &mask);

} // namespace features_internal

/** @brief Computes all valid points in the LiDAR scan [1] Sec. V.A
 * A point can be deemed invalid for 4 reasons
 * 1. The point is at the edge of a scan line
 *    - These points have invalid curvature of (-1) - see compute curvature
 * 2. The point is outside the valid range of the lidar
 *    - These points also invalidate their neighbors as their neighbors
 * curvature will be invalid
 * 3. The point is part of a probably occluded object: There are two cases for
 * this
 *    - Case 1: The current point is a corner occluding something behind
 *        -> i+1 and neighbors to right are invalid
 *        -> The current point is valid since it is probably a corner
 *      ┌─────────────────────────────┐
 *      │───────► Scan Direction      │
 *      │                             │
 *      │  00000000000************    │
 *      │             i+1             │
 *      │                             │
 *      │            i     *=Scan     │
 *      │ ************     0=Occluded │
 *      │                             │
 *      │                             │
 *      │              ^Lidar         │
 *      └─────────────────────────────┘
 *    - Case 2: The current point is part of an occluded object
 *        -> i and neighbors to left are invalid
 *        -> i+1 is valid since it is probably a corner
 *      ┌─────────────────────────────┐
 *      │───────► Scan Direction      │
 *      │                             │
 *      │   ************00000000000   │
 *      │              i              │
 *      │                             │
 *      │ *=Scan        ************  │
 *      │ 0=Occluded    i+1           │
 *      │                             │
 *      │                             │
 *      │              ^Lidar         │
 *      └─────────────────────────────┘
 * 4. The point is on a plane nearly parallel to the LiDAR Beam
 * @param input_scan: The LiDAR scan, organized in row-major order
 * @param lidar_params: The parameters for the lidar that observed input_scan
 * @tparam PointType Template for point type see README.md
 */
template <template <typename> class Accessor = FieldAccessor,
          typename PointType>
std::vector<bool> computeValidPoints(const std::vector<PointType> &input_scan,
                                     const LidarParams &lidar_params,
                                     const FeatureExtractionParams &params) {
  validateLidarScan(input_scan, lidar_params);
  // Allocate mask (with valid flags)
  std::vector<bool> mask(input_scan.size(), true);

  // Structured search (search over each scan line individually over all points
  // [except points on scan line ends]
  for (size_t scan_line_idx = 0; scan_line_idx < lidar_params.num_rows;
       scan_line_idx++) {
    for (size_t line_pt_idx = 0; line_pt_idx < lidar_params.num_columns;
         line_pt_idx++) {
      const size_t idx =
          (scan_line_idx * lidar_params.num_columns) + line_pt_idx;

      // CHECK 1: Due to edge effects, the first and last neighbor_points points
      // of each scan line are invalid
      if (features_internal::markEdgesInvalid(idx, line_pt_idx, lidar_params,
                                              params, mask))
        continue;

      // Get the current point and its two neighbors
      const PointType prev_point = input_scan.at(idx - 1);
      const PointType point = input_scan.at(idx);
      const PointType next_point = input_scan[idx + 1];

      // Compute the range of each point
      const double point_range = pointRange<Accessor>(point);
      const double next_point_range = pointRange<Accessor>(next_point);
      const double prev_point_range = pointRange<Accessor>(prev_point);

      // CHECK 2: Is the point in the valid range of the LiDAR
      if (features_internal::markOutOfRangeInvalid(idx, point_range,
                                                   lidar_params, params, mask))
        continue;
      // CHECK 3: Occlusions
      if (features_internal::markOccludedInvalid(
              idx, point_range, next_point_range, params, mask))
        continue;
      // CHECK 4: Check if the point is on a plane nearly parallel to the LiDAR
      // Beam (no continue b/c last )
      features_internal::markParallelInvalid(idx, prev_point_range, point_range,
                                             next_point_range, params, mask);
    } // end line point search
  } // end scan line search
  return mask;
};

void makeMask(py::module &m);