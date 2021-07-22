cmake_minimum_required(VERSION 3.10)

# set the project name
project(cuda_mmultiply
    VERSION 1.0
    DESCRIPTION "GEMM CUDA"
    LANGUAGES CXX CUDA)
    
# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_COMPILER //opt/nvidia/hpc_sdk/Linux_x86_64/21.5/compilers/bin/nvc++)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

# add the executable
add_executable(cuda_mmultiply
    src/cuda_mmultiply.cu
#    include/mnist_loader.cc
#    include/mnist_loader.h
    )
