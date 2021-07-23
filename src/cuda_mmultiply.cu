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

template<typename T>
__global__
void naive_matrix_multiply(T *A, T *B, T* C, int width, int C_rows, int C_cols)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;   
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // check boundry conditions
  if( row < C_rows && col < C_cols ){
  /*
    // do the multiplication for one row and col
    T value = 0;
    for(int k = 0; k < width; k++){
      value += A[row * width + k] * B[k * C_cols + col];
    }
    // store result
    C[row * C_cols + col] = value;
   */
   C[row * C_cols + col] = row;
   
    
  }
  

}

template<typename T>
void naive_matrix_multiply_cpu(T *A, T *B, T* C, int width, int C_rows, int C_cols){
  for(int i = 0; i < C_rows; i++)
    for(int j = 0; j < C_cols; j++){
      T value = 0.0f;
      for(int k = 0; k < width; k++){
        value += A[i * width + k] * B[k * C_cols + j];
      }
      C[i * C_cols + j] = value;
    }
}

template<typename T>
bool check_equal(T* A1, T* A2, int rows, int cols){
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++){
      if(abs(A1[i * cols + j] - A2[i * cols + j]) > 0.00001){
          return false;
      }
    }
  
  return true;
}


int main(void)
{
    
  // load weights from npy files
  
  xt::xarray<float> matrix_X = xt::load_npy<float>("../data/random_matrix.npy");
  xt::xarray<float> matrix_Y = xt::load_npy<float>("../data/random_input_mat.npy");

  std::cout << "matrix_X SHAPE: " << xt::adapt(matrix_X.shape()) << std::endl;
  std::cout << "matrix_Y SHAPE: " << xt::adapt(matrix_Y.shape()) << std::endl;
  
  unsigned long X_rows = matrix_X.shape()[0];
  unsigned long X_cols = matrix_X.shape()[1];
  
  unsigned long Y_rows = matrix_Y.shape()[0];
  unsigned long Y_cols = matrix_Y.shape()[1];
  
  unsigned long Z_rows = X_rows;
  unsigned long Z_cols = Y_cols;
  
  unsigned long X_size = X_rows * X_cols;
  unsigned long Y_size = Y_rows * Y_cols;
  unsigned long Z_size = Z_rows * Z_cols;
  
  // host copies of X,Y,Z
  float *X = new float[X_size];
  float *Y = new float[Y_size]; 
  float *Z = new float[Z_size];
  float *Z_cpu = new float[Z_size];
  
  // auto data()const: Returns a constant pointer to the underlying array serving as element storage. 
  // The pointer is such that range [data(); data() + size()] is always a valid range, even if the container is empty (data() is not is not dereferenceable in that case)
  
  X = matrix_X.data();
  Y = matrix_Y.data();
  
  // device copies of X, Y, Z
  float *d_X, *d_Y, *d_Z;
  
  // Allocate space for device copies of X, Y, Z
  cudaMalloc((void **)&d_X, X_size);
  cudaMalloc((void **)&d_Y, Y_size);
  cudaMalloc((void **)&d_Z, Z_size);
  
  // Copy a & b from the host to the device
  cudaMemcpy(d_X, &X, X_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y, &Y, Y_size, cudaMemcpyHostToDevice);
  
  // Matrix Multiplication on GPU
  //dim3 dim_grid(Z_cols/COL_TILE_WIDTH, Z_rows/ROW_TILE_WIDTH, 1);
  //dim3 dim_block(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);
  
  dim3 dim_grid(1, 1, 1);
  dim3 dim_block(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);

  naive_matrix_multiply<float><<<dim_grid, dim_block>>>(X, Y, Z, X_cols, Z_rows, Z_cols);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  // Copy result back to the host
  cudaMemcpy(&Z, d_Z, Z_size, cudaMemcpyDeviceToHost);
  
  xt::xarray<double>::shape_type matrix_Z_shape = {Z_rows, Z_cols};
  xt::xarray<float> matrix_Z = xt::adapt(Z, Z_size, xt::acquire_ownership(), matrix_Z_shape);
  std::cout<<"GPU: matrix_Z"<<std::endl;
  std::cout<<matrix_Z<<std::endl;
  std::cout<<"**********************"<<std::endl;

  // Matrix Multiplication on CPU
  naive_matrix_multiply_cpu<float>(X, Y, Z_cpu, X_cols, Z_rows, Z_cols);
  
  xt::xarray<float> matrix_Z_cpu = xt::adapt(Z_cpu, Z_size, xt::acquire_ownership(), matrix_Z_shape);
  std::cout<<"CPU: matrix_Z"<<std::endl;
  std::cout<<matrix_Z_cpu<<std::endl;
  
  
  if(check_equal<float>(Z, Z_cpu, Z_rows, Z_cols))
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;
    

  // Free memory
  cudaFree(d_X);
  cudaFree(d_Y);
  cudaFree(d_Z);
  
  return 0; 
}