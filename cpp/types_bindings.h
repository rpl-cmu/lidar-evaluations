#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "types.h"

namespace py = pybind11;
using namespace pybind11::literals;

inline void makeTypes(py::module &m) {
  py::class_<Stamp>(m, "Stamp")
      .def(py::init<uint32_t, uint32_t>(), py::kw_only(), "sec"_a, "nsec"_a)
      .def_static("from_sec", &Stamp::from_sec)
      .def_static("from_nsec", &Stamp::from_nsec)
      .def("to_sec", &Stamp::to_sec)
      .def("to_nsec", &Stamp::to_nsec)
      .def_readonly("sec", &Stamp::sec)
      .def_readonly("nsec", &Stamp::nsec)
      .def(py::self < py::self)
      .def(py::self > py::self)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(py::self - py::self)
      .def("__repr__", &Stamp::toString)
      .def("__copy__", [](const Stamp &self) { return Stamp(self); })
      .def(
          "__deepcopy__",
          [](const Stamp &self, py::dict) { return Stamp(self); }, "memo"_a);

  // Lidar
  py::class_<Point>(m, "Point")
      .def(py::init<double, double, double, double, uint32_t, uint32_t, uint8_t,
                    uint16_t>(),
           py::kw_only(), "x"_a = 0, "y"_a = 0, "z"_a = 0, "intensity"_a = 0,
           "t"_a = 0, "range"_a = 0, "row"_a = 0, "col"_a = 0)
      .def_readwrite("x", &Point::x)
      .def_readwrite("y", &Point::y)
      .def_readwrite("z", &Point::z)
      .def_readwrite("intensity", &Point::intensity)
      .def_readwrite("range", &Point::range)
      .def_readwrite("t", &Point::t)
      .def_readwrite("row", &Point::row)
      .def_readwrite("col", &Point::col)
      .def("__repr__", &Point::toString);

  py::class_<LidarMeasurement>(m, "LidarMeasurement")
      .def(py::init<Stamp, std::vector<Point>>(), "stamp"_a, "points"_a)
      .def_readonly("stamp", &LidarMeasurement::stamp)
      .def_readonly("points", &LidarMeasurement::points)
      .def("__repr__", &LidarMeasurement::toString);

  py::class_<LidarParams>(m, "LidarParams")
      .def(py::init<int, int, double, double, double>(), py::kw_only(),
           "num_rows"_a, "num_columns"_a, "min_range"_a, "max_range"_a,
           "rate"_a)
      .def_readonly("num_rows", &LidarParams::num_rows)
      .def_readonly("num_columns", &LidarParams::num_columns)
      .def_readonly("min_range", &LidarParams::min_range)
      .def_readonly("max_range", &LidarParams::max_range)
      .def_readonly("rate", &LidarParams::rate)
      .def("__repr__", &LidarParams::toString);

  // Imu
  py::class_<ImuMeasurement>(m, "ImuMeasurement")
      .def(py::init<Stamp, Eigen::Vector3d, Eigen::Vector3d>(), "stamp"_a,
           "gyro"_a, "accel"_a)
      .def_readonly("stamp", &ImuMeasurement::stamp)
      .def_readonly("gyro", &ImuMeasurement::gyro)
      .def_readonly("accel", &ImuMeasurement::accel)
      .def("__repr__", &ImuMeasurement::toString);

  py::class_<ImuParams>(m, "ImuParams")
      .def(py::init<double, double, double, double, double, double,
                    Eigen::Vector3d>(),
           py::kw_only(), "gyro"_a = 1e-5, "accel"_a = 1e-5,
           "gyro_bias"_a = 1e-6, "accel_bias"_a = 1e-6, "bias_init"_a = 1e-7,
           "integration"_a = 1e-7, "gravity"_a = Eigen::Vector3d(0, 0, 9.81))
      .def_static("up", &ImuParams::up)
      .def_static("down", &ImuParams::down)
      .def_readwrite("gyro", &ImuParams::gyro)
      .def_readwrite("accel", &ImuParams::accel)
      .def_readwrite("gyro_bias", &ImuParams::gyro_bias)
      .def_readwrite("accel_bias", &ImuParams::accel_bias)
      .def_readwrite("bias_init", &ImuParams::bias_init)
      .def_readwrite("integration", &ImuParams::integration)
      .def_readwrite("gravity", &ImuParams::gravity)
      .def("__repr__", &ImuParams::toString);

  py::class_<SO3>(m, "SO3")
      .def(py::init<double, double, double, double>(), py::kw_only(), "qx"_a,
           "qy"_a, "qz"_a, "qw"_a)
      .def_readonly("qx", &SO3::qx)
      .def_readonly("qy", &SO3::qy)
      .def_readonly("qz", &SO3::qz)
      .def_readonly("qw", &SO3::qw)
      .def_static("identity", &SO3::identity)
      .def_static("fromMat", &SO3::fromMat)
      .def("inverse", &SO3::inverse)
      .def("log", &SO3::log)
      .def(py::self * py::self)
      .def("__repr__", &SO3::toString)
      .def("__copy__", [](const SO3 &self) { return SO3(self); })
      .def(
          "__deepcopy__", [](const SO3 &self, py::dict) { return SO3(self); },
          "memo"_a);

  py::class_<SE3>(m, "SE3")
      .def(py::init<SO3, Eigen::Vector3d>(), "rot"_a, "trans"_a)
      .def_static("identity", &SE3::identity)
      .def_static("fromMat", &SE3::fromMat)
      .def_readonly("rot", &SE3::rot)
      .def_readonly("trans", &SE3::trans)
      .def("inverse", &SE3::inverse)
      .def(py::self * py::self)
      .def("__repr__", &SE3::toString)
      .def("__copy__", [](const SE3 &self) { return SE3(self); })
      .def(
          "__deepcopy__", [](const SE3 &self, py::dict) { return SE3(self); },
          "memo"_a);
}