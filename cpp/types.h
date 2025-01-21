#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

struct Stamp {
  uint32_t sec;
  uint32_t nsec;

  static Stamp from_sec(double sec) {
    return Stamp{.sec = uint32_t(sec),
                 .nsec = uint32_t((sec - uint32_t(sec)) * 1e9)};
  }

  static Stamp from_nsec(uint64_t nsec) {
    return Stamp{.sec = uint32_t(nsec / 1e9),
                 .nsec = uint32_t(nsec % uint64_t(1e9))};
  }

  uint64_t to_nsec() const { return uint64_t(sec) * uint64_t(1e9) + nsec; }

  double to_sec() const { return double(sec) + double(nsec) * 1e-9; }

  std::string toString() const { return "Stamp(" + toStringBrief() + ")"; }

  std::string toStringBrief() const {
    size_t n_zeros = 9;
    auto nsec_str = std::to_string(nsec);
    auto nsec_str_leading =
        std::string(9 - std::min(n_zeros, nsec_str.length()), '0') + nsec_str;
    return std::to_string(sec) + "." + nsec_str_leading;
  };

  bool operator<(const Stamp &other) const {
    return sec < other.sec || (sec == other.sec && nsec < other.nsec);
  }

  bool operator>(const Stamp &other) const {
    return sec > other.sec || (sec == other.sec && nsec > other.nsec);
  }

  bool operator==(const Stamp &other) const {
    return sec == other.sec && nsec == other.nsec;
  }

  bool operator!=(const Stamp &other) const { return !(*this == other); }

  double operator-(const Stamp &other) const {
    return to_sec() - other.to_sec();
  }
};

struct Point {
  double x;
  double y;
  double z;
  double intensity;
  Stamp t; // in nanoseconds?
  uint32_t range;
  uint8_t row;
  uint16_t col;

  std::string toString() const {
    return "Point(x: " + std::to_string(x) + ", y: " + std::to_string(y) +
           ", z: " + std::to_string(z) +
           ", intensity: " + std::to_string(intensity) +
           ", t: " + std::to_string(t.to_sec()) +
           ", row: " + std::to_string(row) + ", col: " + std::to_string(col) +
           ")";
  }
};

struct LidarMeasurement {
  Stamp stamp;
  std::vector<Point> points;

  LidarMeasurement(Stamp stamp) : stamp(stamp) {}

  LidarMeasurement(Stamp stamp, std::vector<Point> points)
      : stamp(stamp), points(points) {}

  std::string toString() const {
    std::ostringstream oss;
    oss << "LidarMeasurement(stamp: " << stamp.toStringBrief()
        << ", num_points: " << points.size() << ")";
    return oss.str();
  }
};

struct LidarParams {
  // num_rows = num scan lines / channels / rings
  int num_rows;
  int num_columns;
  double min_range;
  double max_range;
  double rate;

  std::string toString() const {
    return "LidarParams(rows: " + std::to_string(num_rows) +
           ", cols: " + std::to_string(num_columns) +
           ", min_range: " + std::to_string(min_range) +
           ", max_range: " + std::to_string(max_range) +
           ", rate: " + std::to_string(rate) + ")";
  };
};

// TODO: I don't know that we'll need these, left here for now
struct SO3 {
  double qx;
  double qy;
  double qz;
  double qw;

  Eigen::Quaterniond toEigen() const {
    return Eigen::Quaterniond(qw, qx, qy, qz);
  }

  static SO3 fromEigen(const Eigen::Quaterniond &q) {
    return SO3{.qx = q.x(), .qy = q.y(), .qz = q.z(), .qw = q.w()};
  }

  static SO3 identity() { return SO3{.qx = 0, .qy = 0, .qz = 0, .qw = 1}; }

  static SO3 fromMat(const Eigen::Matrix3d &R) {
    return fromEigen(Eigen::Quaterniond(R));
  }

  SO3 inverse() const { return SO3{.qx = -qx, .qy = -qy, .qz = -qz, .qw = qw}; }

  SO3 operator*(const SO3 &other) const {
    return fromEigen(toEigen() * other.toEigen());
  }

  Eigen::Vector3d rotate(const Eigen::Vector3d &v) const {
    return toEigen() * v;
  }

  Eigen::Vector3d log() const {
    Eigen::Quaterniond q = toEigen();
    auto axis = Eigen::AngleAxisd(q);
    return axis.angle() * axis.axis();
  }

  std::string toString() const {
    return "SO3(x: " + std::to_string(qx) + ", y: " + std::to_string(qy) +
           ", z: " + std::to_string(qz) + ", w: " + std::to_string(qw) + ")";
  }

  std::string toStringBrief() const {
    return "x: " + std::to_string(qx) + ", y: " + std::to_string(qy) +
           ", z: " + std::to_string(qz) + ", w: " + std::to_string(qw);
  }
};

struct SE3 {
  SO3 rot;
  Eigen::Vector3d trans;

  SE3(SO3 rot, Eigen::Vector3d trans) : rot(rot), trans(trans) {}

  static SE3 identity() {
    return SE3(SO3::identity(), Eigen::Vector3d::Zero());
  }

  static SE3 fromMat(const Eigen::Matrix4d &T) {
    return SE3(SO3::fromMat(T.block<3, 3>(0, 0)), T.block<3, 1>(0, 3));
  }

  SE3 inverse() const {
    const auto inv_rot = rot.inverse();
    return SE3(inv_rot, inv_rot.rotate(-trans));
  }

  SE3 operator*(const SE3 &other) const {
    return SE3(rot * other.rot, rot.rotate(other.trans) + trans);
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << "SE3(rot: [" << rot.toStringBrief() << "], " << "t: ["
        << trans.transpose() << "])";
    return oss.str();
  }
};