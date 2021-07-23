// https://github.com/charitha22/workspace/blob/master/cuda/mm/naive_matrix_multiply.cu

#include <istream>
#include <iostream>
#include <fstream>

#include <stddef.h>
#include <typeinfo>
#include <stdexcept>

#include <math.h>
#include <functional>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xsort.hpp>

#define ROW_TILE_WIDTH 32
#define COL_TILE_WIDTH 32

__global__
void add(int n, float *x, float *x_1)
{
  for (int i = 0; i < n; i++)
    x_1[i] = x[i] + 1.0f;
}

int main(void)
{
    
  // load weights from npy files
  
  xt::xarray<float> matrix_X = xt::load_npy<float>("../data/random_matrix.npy");

  std::cout << "matrix_X SHAPE: " << xt::adapt(matrix_X.shape()) << std::endl;
  
  std::cout << "matrix_X"<< std::endl << matrix_X << std::endl;
  std::cout<<"**********************"<<std::endl;
  
  unsigned long X_rows = matrix_X.shape()[0];
  unsigned long X_cols = matrix_X.shape()[1];  
  unsigned long X_size = X_rows * X_cols;

  
  // host copies of X,Y,Z
  float *X = new float[X_size];
  float *X_1 = new float[X_size];
  
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&X, X_size*sizeof(float));
  cudaMallocManaged(&X_1, X_size*sizeof(float));
  
  //X = matrix_X.data();
  for (int i = 0; i < X_size; i++)
   X[i] = matrix_X[i];
  
  for (int i = 0; i < X_size; i++)
    std::cout << X[i] << ',' ;
  std::cout << std::endl;
  std::cout<<"**********************"<<std::endl;
  
  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(X_size, X, X_1);
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  for (int i = 0; i < X_size; i++)
    std::cout << X_1[i] << ',' ;
  std::cout << std::endl;
  std::cout<<"**********************"<<std::endl;
  
  
  xt::xarray<double>::shape_type matrix_X_shape = {X_rows, X_cols};
  xt::xarray<float> matrix_X_1 = xt::adapt(X_1, X_size, matrix_X_shape);
  
  std::cout << "matrix_X_1"<< std::endl << matrix_X_1 << std::endl;
  std::cout<<"**********************"<<std::endl;
  
  
  // Free memory
  cudaFree(X);
  cudaFree(X_1);
  
  return 0; 
}