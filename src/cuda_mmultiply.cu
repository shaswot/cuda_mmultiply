// https://github.com/charitha22/workspace/blob/master/cuda/mm/naive_matrix_multiply.cu
// ./cuda_mmultiply <image_no>

#include <istream>
#include <iostream>
#include <fstream>
#include <string>

#include <stddef.h>
#include <typeinfo>
#include <stdexcept>

// https://github.com/arpaka/mnist-loader
#include "../include/mnist_loader.h"

#include <math.h>
#include <functional>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xsort.hpp>

// GLOBAL VARIABLES
uint LAYER_WIDTH = 128;
uint MODEL_SEED = 20210702;
    

// Softmax Function
template <class _Tp>
xt::xarray<_Tp> softmax(xt::xarray<_Tp> a)
{
    xt::xarray<_Tp> temp = xt::exp(a);
    xt::xarray<_Tp> sum = xt::sum(temp);
    return temp/sum;
//     return sum;
//     return xt::reduce(function, input, axes)
}

// ReLu Function
template <class _Tp>
xt::xarray<_Tp> relu(xt::xarray<_Tp> a)
{
    return (xt::abs(a)+a)/2;
}
    
    
// GPC_ID to get thread ID values
struct GPC_ID {
    uint t_idx, t_idy, t_idz;
    uint cta_idx, cta_idy, cta_idz;
    uint warp_id, sm_id, grid_id;
};


// https://stackoverflow.com/questions/612328/difference-between-struct-and-typedef-struct-in-c
typedef struct GPC_ID gpc_id; 

// https://forums.developer.nvidia.com/t/any-way-to-know-on-which-sm-a-thread-is-running/19974/15
// https://www.codeproject.com/Articles/15971/Using-Inline-Assembly-in-C-C
__device__ gpc_id get_gpcid(void) {
     
     struct GPC_ID my_id;
     asm("mov.u32 %0, %tid.x;"    : "=r"(my_id.t_idx)    );
     asm("mov.u32 %0, %tid.y;"    : "=r"(my_id.t_idy)    );
     asm("mov.u32 %0, %tid.z;"    : "=r"(my_id.t_idz)    );

     asm("mov.u32 %0, %warpid;" : "=r"(my_id.warp_id) );
     asm("mov.u32 %0, %smid;"   : "=r"(my_id.sm_id)   );
     asm("mov.u32 %0, %gridid;"   : "=r"(my_id.grid_id)   );
     
     asm("mov.u32 %0, %ctaid.x;"  : "=r"(my_id.cta_idx)  );
     asm("mov.u32 %0, %ctaid.y;"  : "=r"(my_id.cta_idy)  );
     asm("mov.u32 %0, %ctaid.z;"  : "=r"(my_id.cta_idz)  );
     
     return my_id;
}


// Matrix Multiplication using CUDA
template<typename T>
__global__
void naive_matrix_multiply(T *A, T *B, T* C, int width, int C_rows, int C_cols, gpc_id* myid)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;   
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  //uint bad_SM = 0;
  //uint bad_thrd = 23;
  //int error = 0.0f;
  
  // check boundry conditions
  if( row < C_rows && col < C_cols ){
    // do the multiplication for one row and col
    T value = 0;
    for(int k = 0; k < width; k++){
      value += A[row * width + k] * B[k * C_cols + col];
    }
    // store result
    C[row * C_cols + col] = value;
    myid[row * C_cols + col] = get_gpcid();
    
    // Inject Error
    /*
    if(myid[row * C_cols + col].sm_id == bad_SM)
        if(myid[row * C_cols + col].t_idy == bad_thrd) //cannot assume all the cores in SM are faulty.
            C[row * C_cols + col] = error;
    */
  }
}

template <class _Tp>
xt::xarray<_Tp> matmul( xt::xarray<_Tp> matrix_X,
                        xt::xarray<_Tp> matrix_Y)
                        // array of bad SMs
                        // array of bad threads
                        // LAYER_WIDTH
                        // Automatic BLOCK SIZE
{
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
  _Tp *X = new _Tp[X_size];
  _Tp *Y = new _Tp[Y_size]; 
  _Tp *Z = new _Tp[Z_size];
  gpc_id *myid = new gpc_id[Z_size];
  
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&X, X_size*sizeof(_Tp));
  cudaMallocManaged(&Y, Y_size*sizeof(_Tp));
  cudaMallocManaged(&Z, Z_size*sizeof(_Tp));
  cudaMallocManaged(&myid, Z_size*sizeof(gpc_id));

  
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
  
  ROW_TILE_WIDTH = 64; //LAYER WIDTH
  COL_TILE_WIDTH = 1; 

  DIM_COL_WIDTH  = 1;
  DIM_ROW_WIDTH = 2;
  
  // Check if enough CUDA cores are allocated
  assert (ROW_TILE_WIDTH*COL_TILE_WIDTH*DIM_COL_WIDTH*DIM_ROW_WIDTH >= LAYER_WIDTH && "Not enough CUDA cores allocated." );
  
  dim3 dim_grid(DIM_COL_WIDTH, DIM_ROW_WIDTH, 1);
  dim3 dim_block(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);

  naive_matrix_multiply<_Tp><<<dim_grid, dim_block>>>(X, Y, Z, X_cols, Z_rows, Z_cols, myid);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  // Convert product matrix to xtensor
  xt::xarray<double>::shape_type matrix_Z_shape = {Z_rows, Z_cols};
  xt::xarray<float> matrix_Z = xt::adapt(Z, Z_size, xt::no_ownership(), matrix_Z_shape);
  
  
  // Log the output of myid
  // https://stackoverflow.com/questions/25918057/how-to-set-a-fixed-width-with-cout
  size_t headerWidths[5] = {std::string("T_IDX  ").size(),
                            std::string("WRP_ID  ").size(),
                            std::string("SM_ID  ").size(),
                            std::string("GRID_ID  ").size(),
                            std::string("CTA_IDX  ").size()
                            };
  // Redirecting output to a file
  // https://stackoverflow.com/questions/10150468/how-to-redirect-cin-and-cout-to-files
  // https://stackoverflow.com/questions/29464578/append-std-output-of-a-function-to-a-file
  
  const std::string cuda_log_file = "../cuda_logfiles/cuda_log-w" + std::to_string(LAYER_WIDTH) + "-" + std::to_string(MODEL_SEED) + ".txt";
  std::ofstream out(cuda_log_file, std::fstream::app);
  std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
  std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!

  std::cout << "T_IDX  T_IDY  T_IDZ  WRP_ID  SM_ID  GRID_ID  CTA_IDX  CTA_IDY  CTA_IDZ"<< std::endl;
  for (int i = 0; i < Z_size; i++){
      std::cout << std::left << std::setw(headerWidths[0]) << myid[i].t_idx;
      std::cout << std::left << std::setw(headerWidths[0]) << myid[i].t_idy;
      std::cout << std::left << std::setw(headerWidths[0]) << myid[i].t_idz;
      std::cout << std::left << std::setw(headerWidths[1]) << myid[i].warp_id;
      std::cout << std::left << std::setw(headerWidths[2]) << myid[i].sm_id;
      std::cout << std::left << std::setw(headerWidths[3]) << myid[i].grid_id;
      std::cout << std::left << std::setw(headerWidths[4]) << myid[i].cta_idx;
      std::cout << std::left << std::setw(headerWidths[4]) << myid[i].cta_idy;
      std::cout << std::left << std::setw(headerWidths[4]) << myid[i].cta_idz;
      std::cout << std::endl;
  }
  std::cout << "***************************" << std::endl;
  
  std::cout.rdbuf(coutbuf); //reset to standard output again
  
    
  // Free memory
  cudaFree(X);
  cudaFree(Y);
  cudaFree(Z);
  cudaFree(myid);
  
  return matrix_Z;
  
}


int main(int argc, char *argv[])
{
 // load weights from npy files
    
    const std::string dense_weights_file = "../data/weights/mnist_dense-w" + std::to_string(LAYER_WIDTH) + "-" + std::to_string(MODEL_SEED) + "_dense_weights.npy";
    const std::string dense_biases_file = "../data/weights/mnist_dense-w" + std::to_string(LAYER_WIDTH) + "-" + std::to_string(MODEL_SEED) + "_dense_biases.npy";
    
    const std::string dense_weights_1_file = "../data/weights/mnist_dense-w" + std::to_string(LAYER_WIDTH) + "-" + std::to_string(MODEL_SEED) + "_dense_1_weights.npy";
    const std::string dense_biases_1_file = "../data/weights/mnist_dense-w" + std::to_string(LAYER_WIDTH) + "-" + std::to_string(MODEL_SEED) + "_dense_1_biases.npy";
    
 
    xt::xarray<float> dense_weights = xt::load_npy<float>(dense_weights_file);
    xt::xarray<float> dense_biases = xt::load_npy<float>(dense_biases_file);
    dense_biases.reshape({-1, 1});
    
    xt::xarray<float> dense_1_weights = xt::load_npy<float>(dense_weights_1_file);
    xt::xarray<float> dense_1_biases = xt::load_npy<float>(dense_biases_1_file);
    dense_1_biases.reshape({-1, 1});


    // load mnist data
    mnist_loader train("../dataset/train-images-idx3-ubyte",
                     "../dataset/train-labels-idx1-ubyte", 60000);
    mnist_loader test("../dataset/t10k-images-idx3-ubyte",
                    "..//dataset/t10k-labels-idx1-ubyte", 10000);


    /*check for the image <image_no> and display truth label*/
    // https://stackoverflow.com/questions/5029840/convert-char-to-int-in-c-and-c
    
   
    int image_no = std::stoi(argv[1]); //convert argument string to int
    int label = train.labels(image_no);
    std::cout << "IMAGE_NUMBER: " << image_no << std::endl;
    std::cout << "TRUTH_LABEL:  " << label << std::endl;
    
    // load the image <image_no> into vector and convert to xtensor<float32>
    std::vector<double> image = train.images(image_no);
        
    // cast to float32 from double and reshape to single batch size
    xt::xarray<float> input_image = xt::adapt(image);
    input_image.reshape({-1, 1});

    /******************LAYER 1******************/
    // transpose weight matrix from (784, LAYER_WIDTH) -> (LAYER_WIDTH,784)
    xt::xarray<float> tr_dense_weights = xt::transpose(dense_weights);
    
    // send through layer
    xt::xarray<float> l1 = matmul<float>(tr_dense_weights, input_image);
    
    // first layer bias
    xt::xarray<float> b1 = l1 + dense_biases;
    
    // relu activation
    xt::xarray<float> b1_relu = relu<float>(b1);

    /******************LAYER 2******************/
    // transpose weight matrix
    xt::xarray<float> tr_dense_1_weights = xt::transpose(dense_1_weights);
    
    // send through layer
    xt::xarray<float> l2 = matmul<float>(tr_dense_1_weights, b1_relu);
    
    // second layer bias
    xt::xarray<float> b2 = l2 + dense_1_biases;
    
    // softmax activation
    xt::xarray<float> l3 = softmax(b2);
    
    // argmax
    std::cout << "PREDICTION:   " << xt::argmax(l3, 0)[0] << std::endl;
    
  return 0; 
}