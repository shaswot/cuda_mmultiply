// https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/

#include <istream>
#include <iostream>
#include <fstream>

#include <stddef.h>
#include <typeinfo>
#include <stdexcept>

// https://github.com/arpaka/mnist-loader
// #include "../include/mnist_loader.h"

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnpy.hpp>


__global__ void matrixMultiplicationKernel(xt::xarray<float> a, 
                                           xt::xarray<float> b,
                                           xt::xarray<float> c,
                                           int K) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;
    // each thread computes one element of the block sub-matrix
    for (int i = 0; i < K; i++) {
        tmpSum += a[ROW * K + i] * b[i * K + COL];
    }

    c[ROW * K + COL] = tmpSum;
}

xt::xarray<float> matrixMultiplication( xt::xarray<float> a, 
                                        xt::xarray<float> b)
{

    const unsigned int n = a.shape()[0]; // a rows
    const unsigned int m = a.shape()[1]; // a cols
    const unsigned int p = b.shape()[1]; // b cols
    
    // array to hold product
    xt::xarray<double>::shape_type shape = {n,p};
    xt::xarray<float> c = xt::zeros<float>(shape);

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(n, p);
    dim3 blocksPerGrid(1, 1);
    
        if (n*p > 512){
            const unsigned int largest_dimension = n > p ? n : p;
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(double(largest_dimension*largest_dimension)/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(largest_dimension*largest_dimension)/double(threadsPerBlock.y));
        }

    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(a, b, c, p);
    
    return c;
}



int main()
{
    auto w_mat = xt::load_npy<float>("../data/random_array.npy");
    auto i_vec = xt::load_npy<float>("../data/random_input.npy");
    auto i_mat = xt::load_npy<float>("../data/random_input_mat.npy");

    i_vec.reshape({-1, 1});
    
    auto y1 = matrixMultiplication(w_mat, i_vec);
    auto y2 = matrixMultiplication(w_mat, i_mat);
    // auto y1 = cuda_matmul<float>(w_mat, i_vec);
    // auto y2 = cuda_matmul<float>(w_mat, i_mat);
    std::cout << "Matrix" << std::endl << w_mat << std::endl;
    std::cout << "Vector" << std::endl << i_vec << std::endl;
    std::cout << "Matrix * Vector" << std::endl << y1 << std::endl;
    std::cout << "******************************" << std::endl;
    
    std::cout << "Matrix A" << std::endl << w_mat << std::endl;
    std::cout << "Matrix B" << std::endl << i_mat << std::endl;
    std::cout << "Matrix * Matrix" << std::endl << y2 << std::endl;

    
    return 0;
}