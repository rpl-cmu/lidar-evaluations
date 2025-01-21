#################### BASIC SETUP ####################
cmake_minimum_required(VERSION 3.16)
project(LIDAR_FEATURES VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(FetchContent)

#################### IMPORT DEPENDENCIES ####################
# Eigen 
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    FIND_PACKAGE_ARGS NO_MODULE
)
FetchContent_MakeAvailable(Eigen3)

# pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(pybind11)

# nanoflann
FetchContent_Declare(
    nanoflann
    GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
    GIT_TAG        v1.6.3
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(nanoflann)


#################### ADD TARGETS ####################
# Build python bindings
pybind11_add_module(_cpp cpp/bindings.cpp cpp/helpers.cpp)
target_link_libraries(_cpp PUBLIC Eigen3::Eigen nanoflann::nanoflann)
install(TARGETS _cpp DESTINATION features)