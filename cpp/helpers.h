#pragma once
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "types.h"

namespace py = pybind11;
using namespace pybind11::literals;

// ------------------- Conversion from ros pointcloud2 ------------------- //
enum DataType {
  INT8 = 1,
  UINT8 = 2,
  INT16 = 3,
  UINT16 = 4,
  INT32 = 5,
  UINT32 = 6,
  FLOAT32 = 7,
  FLOAT64 = 8,
};

struct Field {
  std::string name;
  DataType datatype;
  uint32_t offset;
};

struct PointCloudMetadata {
  Stamp stamp;
  int width;
  int height;
  int point_step;
  int row_step;
  int is_bigendian;
  int is_dense;
};

template <typename T>
std::function<void(T &, const uint8_t *)> data_getter(DataType datatype,
                                                      const uint32_t offset) {
  switch (datatype) {
  case UINT8: {
    return [offset](T &value, const uint8_t *data) noexcept {
      value = static_cast<T>(*reinterpret_cast<const uint8_t *>(data + offset));
    };
  }
  case INT8: {
    return [offset](T &value, const uint8_t *data) noexcept {
      value = static_cast<T>(*reinterpret_cast<const int8_t *>(data + offset));
    };
  }
  case UINT16: {
    return [offset](T &value, const uint8_t *data) noexcept {
      value =
          static_cast<T>(*reinterpret_cast<const uint16_t *>(data + offset));
    };
  }
  case UINT32: {
    return [offset](T &value, const uint8_t *data) noexcept {
      value =
          static_cast<T>(*reinterpret_cast<const uint32_t *>(data + offset));
    };
  }
  case INT16: {
    return [offset](T &value, const uint8_t *data) noexcept {
      value = static_cast<T>(*reinterpret_cast<const int16_t *>(data + offset));
    };
  }
  case INT32: {
    return [offset](T &value, const uint8_t *data) noexcept {
      value = static_cast<T>(*reinterpret_cast<const int32_t *>(data + offset));
    };
  }
  case FLOAT32: {
    return [offset](T &value, const uint8_t *data) noexcept {
      value = static_cast<T>(*reinterpret_cast<const float *>(data + offset));
    };
  }
  case FLOAT64: {
    return [offset](T &value, const uint8_t *data) noexcept {
      value = static_cast<T>(*reinterpret_cast<const double *>(data + offset));
    };
  }
  default: {
    throw std::runtime_error("Unsupported datatype");
  }
  }
}

// Specialization for Stamp
std::function<void(Stamp &, const uint8_t *)>
data_getter(DataType datatype, const uint32_t offset);

template <typename T> std::function<void(T &, const uint8_t *)> blank() {
  return [](T &, const uint8_t *) noexcept {};
}

LidarMeasurement ros_pc2_conversion(const PointCloudMetadata &msg,
                                    const std::vector<Field> &fields,
                                    const uint8_t *data);

// ----------------------- Accessors for nanoflann ----------------------- //
/// @brief Accessor struct for points that store their position as public fields
// Assumes that fields are named x, y, and z, example: PCL Point Type
template <typename PointType> struct FieldAccessor {
  static double x(PointType pt) { return pt.x; }
  static double y(PointType pt) { return pt.y; }
  static double z(PointType pt) { return pt.z; }
};

/// @brief Accessor struct for points that store their positions in a
/// vector/array/matrix accessed with brackets Assumes that order is [x, y, z]
/// and zero indexed, example: Eigen Matrix
template <typename PointType> struct ParenAccessor {
  static double x(PointType pt) { return pt(0); }
  static double y(PointType pt) { return pt(1); }
  static double z(PointType pt) { return pt(2); }
};

/// @brief Accessor struct for points that store their positions in a
/// vector/array/matrix accessed with an `at` func Assumes that order is [x, y,
/// z] and zero indexed, example: std::vector
template <typename PointType> struct AtAccessor {
  static double x(PointType pt) { return pt.at(0); }
  static double y(PointType pt) { return pt.at(1); }
  static double z(PointType pt) { return pt.at(2); }
};

/// @brief Computes the range from the LiDAR to the point
template <template <typename> class Accessor = FieldAccessor,
          typename PointType>
double pointRange(const PointType &pt) {
  return std::sqrt(Accessor<PointType>::x(pt) * Accessor<PointType>::x(pt)   //
                   + Accessor<PointType>::y(pt) * Accessor<PointType>::y(pt) //
                   + Accessor<PointType>::z(pt) * Accessor<PointType>::z(pt));
}

/// @brief Converts a point type into an Eigen 3d vector
template <template <typename> class Accessor = FieldAccessor,
          typename PointType>
Eigen::Vector3d pointToEigen(const PointType &pt) {
  return (Eigen::Vector3d() << Accessor<PointType>::x(pt),
          Accessor<PointType>::y(pt), Accessor<PointType>::z(pt))
      .finished();
}

// TODO: Not sure how valid this'll be for me either
template <typename PointType, template <typename> class Alloc>
void validateLidarScan(
    const std::vector<PointType, Alloc<PointType>> &input_scan,
    const LidarParams &lidar_params) {
  if (input_scan.size() != lidar_params.num_rows * lidar_params.num_columns) {
    std::stringstream msg_stream;
    msg_stream << "LOAM: provided lidar scan size ( " << input_scan.size()
               << ")  does not match provided lidar parameters ("
               << lidar_params.num_rows << " x " << lidar_params.num_columns
               << ")";
    throw std::runtime_error(msg_stream.str());
  }
}

// ---------------------- Create python bindings ---------------------- //
void makeConversions(py::module &m);