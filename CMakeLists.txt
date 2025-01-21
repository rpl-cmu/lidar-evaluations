#################### BASIC SETUP ####################
cmake_minimum_required(VERSION 3.16)
project(LIDAR_FEATURES VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(FetchContent)

#################### IMPORT DEPENDENCIES ####################
# Eigen 
# FetchContent_Declare(
#     Eigen3
#     GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
#     GIT_TAG        3.4.0
#     FIND_PACKAGE_ARGS 3.4 EXACT NO_MODULE
# )
# FetchContent_MakeAvailable(Eigen3)

# gtsam
set(GTSAM_BUILD_TESTS OFF CACHE BOOL "")
set(GTSAM_BUILD_DOCS OFF CACHE BOOL "")
set(GTSAM_BUILD_PYTHON OFF CACHE BOOL "")
set(GTSAM_INSTALL_MATLAB_TOOLBOX OFF CACHE BOOL "")
set(GTSAM_BUILD_UNSTABLE OFF CACHE BOOL "")
set(GTSAM_BUILD_EXAMPLES OFF CACHE BOOL "")
set(GTSAM_BUILD_EXAMPLES_ALWAYS OFF CACHE BOOL "")
set(GTSAM_USE_SYSTEM_EIGEN OFF CACHE BOOL "")
FetchContent_Declare(
    GTSAM
    GIT_REPOSITORY https://github.com/borglab/gtsam.git
    GIT_TAG        4.2.0
    FIND_PACKAGE_ARGS 4.2 EXACT
)
FetchContent_MakeAvailable(GTSAM)
# disable loud gtsam errors
target_compile_options(gtsam PRIVATE -Wno-deprecated-declarations)
target_compile_options(gtsam PRIVATE -Wno-unused-but-set-variable)
target_compile_options(gtsam PRIVATE -Wno-cpp)

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
pybind11_add_module(_cpp cpp/bindings.cpp cpp/helpers.cpp cpp/types.cpp)
target_link_libraries(_cpp PUBLIC Eigen3::Eigen nanoflann::nanoflann gtsam)
install(TARGETS _cpp DESTINATION features)