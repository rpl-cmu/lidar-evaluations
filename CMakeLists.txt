#################### BASIC SETUP ####################
cmake_minimum_required(VERSION 3.16)
project(LIDAR_FEATURES VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#################### IMPORT DEPENDENCIES ####################
set(LOAM_BUILD_PYTHON ON CACHE BOOL "Build loam python bindings" FORCE)
add_subdirectory(loam)

set(EVALIO_BUILD_ALL OFF CACHE BOOL "Don't build included pipelines" FORCE)
set(EVALIO_KISS_ICP OFF CACHE BOOL "Don't build KISS-ICP" FORCE)
set(EVALIO_LIO_SAM OFF CACHE BOOL "Don't build LIO-SAM" FORCE)
add_subdirectory(evalio)

#################### ADD TARGETS ####################
# Build python bindings
pybind11_add_module(_cpp cpp/bindings.cpp cpp/helpers.cpp cpp/types.cpp)
target_link_libraries(_cpp PUBLIC Eigen3::Eigen nanoflann::nanoflann gtsam)
install(TARGETS _cpp DESTINATION features)