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

#include <curand.h>

#define BLOCK_HEIGHT 16
#define BLOCK_WIDTH 196

// GLOBAL VARIABLES
uint LAYER_WIDTH = 32;
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
__device__ gpc_id get_gpcid(void) 
{
     
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

template<typename T>
__global__ void zero_vector(T *vec, const int n)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < n )
    vec[xIndex]=0.0f;
}

// // Matrix-vector multiplication using CUDA
// // Using shared memory and avoiding banking conflicts
template<typename T>
__global__ void MatMulKernel(T *out, T *in, T *a, const int matrixHeight, const int matrixWidth) {
  // get variables for loop
  // copy section of b into shared mem
  // go through the threads vertically and sum them into a variable
  // atomic add these variables to the corresponding c index

  // looping is happening horizontally on the matrix
  // BLOCK_WIDTH is again horizontal
  // BLOCK_HEIGHT is going vertical
  // n_cols / BLOCK_WIDTH blocks horizontally
  // n_rows / BLOCK_HEIGHT block vertically

  // get variables for loop
  // variable for loop length: blockEltHeight
  __shared__ int blockElt;
  __shared__ int blockxInd;
  __shared__ int blockyInd;
  if (threadIdx.x == 0) // only the first thread of the entire block initializes the shared variables blockElt, blockxInd, blockyInd.
  {  
    if ((blockIdx.x + 1) * BLOCK_WIDTH <= matrixWidth)
      blockElt = BLOCK_WIDTH; // NOT the rightmost block so width of block = BLOCK_WIDTH
    else blockElt = matrixWidth % BLOCK_WIDTH; // rightmost block so width of block = matrixWidth % BLOCK_WIDTH
    blockxInd = blockIdx.x * BLOCK_WIDTH; // top left thread x-index of the block
    blockyInd = blockIdx.y * BLOCK_HEIGHT; // top left thread y-index of the block
  }
  
  __syncthreads(); //all threads have value of blockElt, blockxInd, blockyInd
  
  // copy section of b into shared mem
  // https://stackoverflow.com/questions/24419822/efficiently-initializing-shared-memory-array-in-cuda/24419969#24419969
  // use threads to write into independent locations of b[] from in [] 
  __shared__ T b[BLOCK_WIDTH];
  
  int threads_per_block = BLOCK_HEIGHT;
  int lidx = threadIdx.x;
  while (lidx < BLOCK_WIDTH)
  {
    b[lidx] = in[lidx + blockIdx.x * BLOCK_WIDTH];
    lidx += threads_per_block;
  }  
  __syncthreads();
  
   
  // summing variable
  T cSum = (T) 0.0;
  int threadyInd = blockyInd + threadIdx.x;

  // make sure we are inside the matrix verticallly
  if (threadyInd < matrixHeight) 
  {
    // go through the threads vertically and sum them into a variable
    
    #pragma unroll
    for (int i=0; i<blockElt; i++)
    {
      // row R of matrix a[] --> blockIdx.y * BLOCK_HEIGHT + threadIdx.x) * matrixWidth 
      // col C of row R of matrix a[] --> blockIdx.x * BLOCK_WIDTH 
      // element E of col C of row R of matrix a[] --> i
      
      cSum += b[i] * a[(blockIdx.y * BLOCK_HEIGHT + threadIdx.x) * matrixWidth + blockIdx.x * BLOCK_WIDTH + i];
//       if (i==blockElt-1)
//       printf("blockxInd = %d, blockyInd = %d, threadIdx.x = %d, csum = %f\n", blockxInd, blockyInd, threadIdx.x, cSum);
    }
    // atomic add these variables to the corresponding c index
    atomicAdd(out + threadyInd, cSum);
  }
  
}

template <class _Tp>
xt::xarray<_Tp> matVecMul (xt::xarray<_Tp> matrix_A, 
                           xt::xarray<_Tp> vector_B)
{
  unsigned int n_rows = matrix_A.shape()[0];
  unsigned int n_cols = matrix_A.shape()[1];
  
  unsigned int size_A = n_rows * n_cols;
  unsigned int size_B = n_cols;
  assert (vector_B.shape()[0] == size_B && "matrix A and vector B shape mismatch.");
  assert (vector_B.shape()[1] == 1 && "vector B no. of columns != 1");
  unsigned int size_C = n_rows;
  
  // declare matrices for GPU and allocate memory
  
  // host copies of A,B,C
  _Tp *A = new _Tp[size_A];
  _Tp *B = new _Tp[size_B]; 
  _Tp *C = new _Tp[size_C];
  
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&A, size_A*sizeof(_Tp));
  cudaMallocManaged(&B, size_B*sizeof(_Tp));
  cudaMallocManaged(&C, size_C*sizeof(_Tp));
  
  // Fill the matrix values from xtensor to C++ array
  for (int i = 0; i < size_A; i++)
  A[i] = matrix_A.flat(i);
   
  for (int i = 0; i < size_B; i++)
  B[i] = vector_B.flat(i);
  

  //run mat-vec multiplication
  // set up threading and blocking variables
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  unsigned int max_threads_per_block = dp.maxThreadsPerBlock;
  
  assert(BLOCK_HEIGHT <= max_threads_per_block && "BLOCK_HEIGHT exceeds max_threads_per_block");
  
  // Block Grid for zero_vector_float<<< >>>
  int threads_perblockm = min(n_rows, max_threads_per_block);
  dim3 threadsPerBlockm(threads_perblockm);
  int num_blocksm = (int)ceil((float)n_rows/(float)threads_perblockm);
  dim3 numBlocksm(num_blocksm);

  // Block Grid for MatMulKernel<<< >>>
  int blockCols = (int) ceil(n_cols / (double) BLOCK_WIDTH);
  int blockRows = (int) ceil(n_rows / (double) BLOCK_HEIGHT);
  dim3 dimBlock(BLOCK_HEIGHT); // BLOCK_HEIGHT directly corresponds to no. of threads per block i.e., one thread per row of the block.
  dim3 dimGrid(blockCols, blockRows);
  std::cout << "Gridblock size: (" << blockCols << ","<< blockRows << ")" << std::endl;

  int sharedMem = 3 * sizeof (int) + BLOCK_WIDTH * sizeof(_Tp);
  // 3 * sizeof (int) -> to store blockElt, blockxInd, blockyInd;

  // execute kernels
  zero_vector<float><<<numBlocksm, threadsPerBlockm>>>(C, n_rows);
  MatMulKernel<float><<<dimGrid, dimBlock, sharedMem>>>(C, B, A, n_rows, n_cols);

  cudaDeviceSynchronize();
   
  // Convert product vector to xtensor
  xt::xarray<double>::shape_type C_shape = {size_C, 1};
  xt::xarray<_Tp> vec_C = xt::adapt(C, size_C, xt::no_ownership(), C_shape);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  
  return vec_C;
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
    
  //combine test and train images
  std::vector<std::vector<double>> all_images;
  int all_labels[70000];
    
  for (int i = 0; i < 60000; i++)
  {
      all_images.push_back(train.images(i));
      all_labels[i] = train.labels(i);
  }
    
  for (int i = 0; i < 10000; i++)
  {
      all_images.push_back(test.images(i));
      all_labels[60000+i] = test.labels(i);
  }


  /*check for the image <image_no> and display truth label*/
  // https://stackoverflow.com/questions/5029840/convert-char-to-int-in-c-and-c   
  int image_no = std::stoi(argv[1]); //convert argument string to int
  int label = all_labels[image_no];
  std::cout << "IMAGE_NUMBER: " << image_no << std::endl;
  std::cout << "TRUTH_LABEL:  " << label << std::endl;
    
  // load the image <image_no> into vector and convert to xtensor<float32>
  std::vector<double> image = all_images[image_no];
        
  // cast to float32 from double and reshape to single batch size
  xt::xarray<float> input_image = xt::adapt(image);
  input_image.reshape({-1, 1});
  
  /******************LAYER 1******************/
  // transpose weight matrix from (784, LAYER_WIDTH) -> (LAYER_WIDTH,784)
  xt::xarray<float> tr_dense_weights = xt::transpose(dense_weights);
    
  // send through layer
  xt::xarray<float> l1 = matVecMul(tr_dense_weights, input_image);
  
//   std::cout << "Transposed Weight Matrix" << std::endl << tr_dense_weights << std::endl;
//   std::cout << "Transposed Weight Matrix Shape"<<xt::adapt(tr_dense_weights.shape())<< std::endl;
//   std::cout << "***************************************" << std::endl;
//   std::cout << "Input Vector" << std::endl << input_image << std::endl;
//   std::cout << "***************************************" << std::endl;
//   std::cout << "Output Vector" << std::endl << l1 << std::endl;
//   std::cout << "***************************************" << std::endl;


  // first layer bias
  xt::xarray<float> b1 = l1 + dense_biases;
    
  // relu activation
  xt::xarray<float> b1_relu = relu<float>(b1);

  /******************LAYER 2******************/
  // transpose weight matrix
  xt::xarray<float> tr_dense_1_weights = xt::transpose(dense_1_weights);
    
  // send through layer
  xt::xarray<float> l2 = matVecMul<float>(tr_dense_1_weights, b1_relu);
    
  // second layer bias
  xt::xarray<float> b2 = l2 + dense_1_biases;
    
  // softmax activation
  xt::xarray<float> l3 = softmax(b2);
    
  // argmax
  std::cout << "PREDICTION:   " << xt::argmax(l3, 0)[0] << std::endl;
    
  return 0; 
    
    
    /*********************************************************************************/
//  // load weights from npy files
    
//     const std::string dense_weights_file = "../data/weights/mnist_dense-w" + std::to_string(LAYER_WIDTH) + "-" + std::to_string(MODEL_SEED) + "_dense_weights.npy";
//     const std::string dense_biases_file = "../data/weights/mnist_dense-w" + std::to_string(LAYER_WIDTH) + "-" + std::to_string(MODEL_SEED) + "_dense_biases.npy";
    
//     const std::string dense_weights_1_file = "../data/weights/mnist_dense-w" + std::to_string(LAYER_WIDTH) + "-" + std::to_string(MODEL_SEED) + "_dense_1_weights.npy";
//     const std::string dense_biases_1_file = "../data/weights/mnist_dense-w" + std::to_string(LAYER_WIDTH) + "-" + std::to_string(MODEL_SEED) + "_dense_1_biases.npy";
 
//     xt::xarray<float> dense_weights = xt::load_npy<float>(dense_weights_file);
//     xt::xarray<float> dense_biases = xt::load_npy<float>(dense_biases_file);
//     dense_biases.reshape({-1, 1});
    
//     xt::xarray<float> dense_1_weights = xt::load_npy<float>(dense_weights_1_file);
//     xt::xarray<float> dense_1_biases = xt::load_npy<float>(dense_biases_1_file);
//     dense_1_biases.reshape({-1, 1});


//     // load mnist data
//     mnist_loader train("../dataset/train-images-idx3-ubyte",
//                      "../dataset/train-labels-idx1-ubyte", 60000);
//     mnist_loader test("../dataset/t10k-images-idx3-ubyte",
//                     "..//dataset/t10k-labels-idx1-ubyte", 10000);
    
//     //combine test and train images
//     std::vector<std::vector<double>> all_images;
//     int all_labels[70000];
    
//     for (int i = 0; i < 60000; i++)
//     {
//         all_images.push_back(train.images(i));
//         all_labels[i] = train.labels(i);
//     }
    
//     for (int i = 0; i < 10000; i++)
//     {
//         all_images.push_back(test.images(i));
//         all_labels[60000+i] = test.labels(i);
//     }


//     /*check for the image <image_no> and display truth label*/
//     // https://stackoverflow.com/questions/5029840/convert-char-to-int-in-c-and-c   
//     int image_no = std::stoi(argv[1]); //convert argument string to int
//     int label = all_labels[image_no];
//     std::cout << "IMAGE_NUMBER: " << image_no << std::endl;
//     std::cout << "TRUTH_LABEL:  " << label << std::endl;
    
//     // load the image <image_no> into vector and convert to xtensor<float32>
//      std::vector<double> image = all_images[image_no];
        
//     // cast to float32 from double and reshape to single batch size
//     xt::xarray<float> input_image = xt::adapt(image);
//     input_image.reshape({-1, 1});

//     /******************LAYER 1******************/
//     // transpose weight matrix from (784, LAYER_WIDTH) -> (LAYER_WIDTH,784)
//     xt::xarray<float> tr_dense_weights = xt::transpose(dense_weights);
    
//     // send through layer
//     xt::xarray<float> l1 = matVecMul<float>(tr_dense_weights, input_image);
    
//     // first layer bias
//     xt::xarray<float> b1 = l1 + dense_biases;
    
//     // relu activation
//     xt::xarray<float> b1_relu = relu<float>(b1);

//     /******************LAYER 2******************/
//     // transpose weight matrix
//     xt::xarray<float> tr_dense_1_weights = xt::transpose(dense_1_weights);
    
//     // send through layer
//     xt::xarray<float> l2 = matVecMul<float>(tr_dense_1_weights, b1_relu);
    
//     // second layer bias
//     xt::xarray<float> b2 = l2 + dense_1_biases;
    
//     // softmax activation
//     xt::xarray<float> l3 = softmax(b2);
    
//     // argmax
//     std::cout << "PREDICTION:   " << xt::argmax(l3, 0)[0] << std::endl;
    
//   return 0; 
}