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

// https://forums.developer.nvidia.com/t/any-way-to-know-on-which-sm-a-thread-is-running/19974/15
/* E.D. Riedijk */

__device__ uint get_smid(void) {
     uint ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}

/*
__global__ void kern(int *sm){
   if (threadIdx.x==0)
      sm[blockIdx.x]=get_smid();
}
*/

template<typename T>
__global__
void naive_matrix_multiply(T *A, T *B, T* C, int width, int C_rows, int C_cols, int* sm)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;   
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // check boundry conditions
  if( row < C_rows && col < C_cols ){
    // do the multiplication for one row and col
    T value = 0;
    for(int k = 0; k < width; k++){
      value += A[row * width + k] * B[k * C_cols + col];
    }
    // store result
    C[row * C_cols + col] = value;    
  }
  sm[row * C_cols + col]=get_smid();
  

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
  
  //xt::xarray<float> matrix_X = xt::load_npy<float>("../data/random_matrix.npy");
  //xt::xarray<float> matrix_Y = xt::load_npy<float>("../data/random_input_mat.npy");
  
  xt::xarray<float> dense_weights = xt::load_npy<float>("../data/dense_weights.npy");
  xt::xarray<float> matrix_X = xt::transpose(dense_weights);
  
  xt::xarray<float> matrix_Y = xt::load_npy<float>("../data/image_65.npy");
  matrix_Y.reshape({-1, 1});

  
  std::cout << "matrix_X SHAPE: " << xt::adapt(matrix_X.shape()) << std::endl;
  std::cout << "matrix_Y SHAPE: " << xt::adapt(matrix_Y.shape()) << std::endl;
  /*
  std::cout << matrix_X << std::endl;
  std::cout<<"**********************"<<std::endl;
  std::cout << matrix_Y << std::endl;
  std::cout<<"**********************"<<std::endl;
  */
  
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
  int *sm = new int[Z_size];
  
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&X, X_size*sizeof(float));
  cudaMallocManaged(&Y, Y_size*sizeof(float));
  cudaMallocManaged(&Z, Z_size*sizeof(float));
  cudaMallocManaged(&sm, Z_size*sizeof(int));

  
  // Fill the matrix values from xtensor to C++ array
  for (int i = 0; i < X_size; i++)
  X[i] = matrix_X.flat(i);
   
  for (int i = 0; i < Y_size; i++)
  Y[i] = matrix_Y.flat(i);
  
  
  // Matrix Multiplication on GPU
  // Determine Grid/Block Size
  // https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
  unsigned long ROW_TILE_WIDTH, COL_TILE_WIDTH;
  unsigned long DIM_COL_WIDTH, DIM_ROW_WIDTH;
  unsigned long LARGEST_COL_DIM, LARGEST_ROW_DIM;
  /*
  LARGEST_COL_DIM = X_cols > Y_cols ? X_cols : Y_cols;
  LARGEST_COL_DIM = LARGEST_COL_DIM > Z_cols ? LARGEST_COL_DIM : Z_cols;
  
  LARGEST_ROW_DIM = X_rows > Y_rows ? X_rows : Y_rows;
  LARGEST_ROW_DIM = LARGEST_ROW_DIM > Z_rows ? LARGEST_ROW_DIM : Z_rows;
  
  DIM_COL_WIDTH = (LARGEST_COL_DIM + COL_TILE_WIDTH -1)/COL_TILE_WIDTH;
  DIM_ROW_WIDTH = (LARGEST_ROW_DIM + ROW_TILE_WIDTH -1)/ROW_TILE_WIDTH;
  */
  
  ROW_TILE_WIDTH = 32; 
  COL_TILE_WIDTH = 1; 

  DIM_COL_WIDTH  = 1;
  DIM_ROW_WIDTH = 1;
  
  dim3 dim_grid(DIM_COL_WIDTH, DIM_ROW_WIDTH, 1);
  dim3 dim_block(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);
  
  std::cout << "GRID SIZE: " << DIM_COL_WIDTH << " , " << DIM_ROW_WIDTH << std::endl;
  std::cout << "BLCK SIZE: " << COL_TILE_WIDTH << " , " << ROW_TILE_WIDTH << std::endl;


  naive_matrix_multiply<float><<<dim_grid, dim_block>>>(X, Y, Z, X_cols, Z_rows, Z_cols, sm);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  // Convert product matrix to xtensor
  xt::xarray<double>::shape_type matrix_Z_shape = {Z_rows, Z_cols};
  xt::xarray<float> matrix_Z = xt::adapt(Z, Z_size, matrix_Z_shape);
  std::cout<<"GPU: matrix_Z"<<std::endl;
  std::cout<<matrix_Z<<std::endl;
  std::cout<<"**********************"<<std::endl;

  // Matrix Multiplication on CPU
  naive_matrix_multiply_cpu<float>(X, Y, Z_cpu, X_cols, Z_rows, Z_cols);
  
  xt::xarray<float> matrix_Z_cpu = xt::adapt(Z_cpu, Z_size, matrix_Z_shape);
  std::cout<<"CPU: matrix_Z"<<std::endl;
  std::cout<<matrix_Z_cpu<<std::endl;
  
  
  if(check_equal<float>(Z, Z_cpu, Z_rows, Z_cols))
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;
    
// Convert product matrix to xtensor
  xt::xarray<int> SM_ID = xt::adapt(sm, Z_size, matrix_Z_shape);
  std::cout<<"SM ID"<<std::endl;
  std::cout<<SM_ID<<std::endl;
  std::cout<<"**********************"<<std::endl;
    

  // Free memory
  cudaFree(X);
  cudaFree(Y);
  cudaFree(Z);
  
  return 0; 
}