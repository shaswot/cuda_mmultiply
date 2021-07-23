cmake_minimum_required(VERSION 3.10)

# set the project name
project(cuda_mmultiply
    VERSION 1.0
    DESCRIPTION "GEMM CUDA"
    LANGUAGES CXX CUDA)
    
# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_COMPILER nvc++)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

# add the executable
add_executable(cuda_mmultiply
    src/cuda_mmultiply.cu
#    include/mnist_loader.cc
#    include/mnist_loader.h
    )
