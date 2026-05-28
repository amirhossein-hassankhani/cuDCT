cmake_minimum_required(VERSION 3.24)

project(cuDCT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ============================================================
# CUDA core static library
# ============================================================

add_library(cuDCT_core STATIC
    src/DCT1D.cu
    src/DCT2D.cu
    src/DCT3D.cu
)

target_include_directories(cuDCT_core PUBLIC
    src
)

target_link_libraries(cuDCT_core PUBLIC
    CUDA::cudart
    CUDA::cufft
)

set_target_properties(cuDCT_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================
# Pybind extension module
# ============================================================

pybind11_add_module(_cuDCT
    src/bindings.cpp
)

target_include_directories(_cuDCT PRIVATE
    src
    extern/dlpack/include
)

target_link_libraries(_cuDCT PRIVATE
    cuDCT_core
    CUDA::cudart
    CUDA::cufft
)

set_target_properties(_cuDCT PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

install(TARGETS _cuDCT
    LIBRARY DESTINATION cuDCT
)