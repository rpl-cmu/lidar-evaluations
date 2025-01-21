#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "conversions.h"
#include "types_bindings.h"

PYBIND11_MODULE(_cpp, m) {
  auto m_types = m.def_submodule(
      "types",
      "Common types used for conversion between datasets and pipelines.");
  makeTypes(m_types);

  auto m_conversion = m.def_submodule(
      "_conversion", "Helper functions for internal evalio usage.");
  makeConversions(m_conversion);
}
