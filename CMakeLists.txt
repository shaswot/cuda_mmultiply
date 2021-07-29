cmake_minimum_required(VERSION 3.10)

# specify CUDA architecture
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# https://cmake.org/cmake/help/git-stage/prop_tgt/CUDA_ARCHITECTURES.html

# Volta, RTX 2080, 2080Ti, 3080
set(CMAKE_CUDA_ARCHITECTURES "70;75;80")

# set the project name
project(cuda_mmultiply
    VERSION 1.0
    DESCRIPTION "GEMM CUDA"
    LANGUAGES CXX CUDA)
    
# specify the C++ standard
# C17 is not compatible because xtl/xtensor were installed using cxx_std_14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_COMPILER nvc++)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)




# add the executable
add_executable(cuda_mmultiply
    src/cuda_mmultiply.cu
    include/mnist_loader.cc
    include/mnist_loader.h
    )
