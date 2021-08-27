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

#define BLOCK_HEIGHT 32
#define BLOCK_WIDTH 392

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

void createRandomMatrix(float *A, int size, int seed) {
  float *d_A;
  float *h_A = (float *) malloc (size * sizeof(float));
  curandGenerator_t gen;
  size_t size_d_A = size * sizeof(float);

  cudaMalloc((void **) &d_A, size_d_A);

  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  curandGenerateUniform(gen, d_A, size);

  cudaMemcpy(h_A, d_A, size_d_A, cudaMemcpyDeviceToHost);

  // for (int j = 0; j < 10; j++) 
  //  printf("h_A[%d] = %l=f\n", j, 10* h_A[j]);

  for (int j = 0; j < size; j++) 
    A[j] = h_A[j] / sqrt (size); 

  curandDestroyGenerator(gen);
  cudaFree(d_A);
  free(h_A);
}

__global__ void zero_vector_float(float *vec, const int n)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < n )
    vec[xIndex]=0.0f;
}

// // Matrix-vector multiplication using CUDA
// // Using shared memory and avoiding banking conflicts
// xt::xarray<_Tp> matvecmul( xt::xarray<_Tp> matrix_W,
//                         xt::xarray<_Tp> matrix_X)
__global__ void MatMulKernel(float *out, float *in, float *a, const int matrixHeight, const int matrixWidth) {
  // get variables for loop
  // copy section of b into shared mem
  // go through the threads vertically and sum them into a variable
  // atomic add these variables to the corresponding c index

  // looping is happening horizontally on the matrix
  // BLOCK_WIDTH is again horizontal
  // BLOCK_HEIGHT is going vertical
  // n / BLOCK_WIDTH blocks horizontally
  // m / BLOCK_HEIGHT block vertically

  // get variables for loop
  // variable for loop length: blockEltHeight
  __shared__ int blockElt;
  __shared__ int blockxInd;
  __shared__ int blockyInd;
  if (threadIdx.x == 0) {
    if ((blockIdx.x + 1) * BLOCK_WIDTH <= matrixWidth)
      blockElt = BLOCK_WIDTH;
    else blockElt = matrixWidth % BLOCK_WIDTH;
    blockxInd = blockIdx.x * BLOCK_WIDTH;
    blockyInd = blockIdx.y * BLOCK_HEIGHT;
  }
  
  __syncthreads();
  
  // copy section of b into shared mem
  // use the first BLOCK_WIDTH of thread
  __shared__ float b[BLOCK_WIDTH];

  if (threadIdx.x < blockElt) 
    b[threadIdx.x] = in[blockxInd + threadIdx.x];
  
  __syncthreads();

  // summing variable
  float cSum = (float) 0;
  int threadyInd = blockyInd + threadIdx.x;

  // make sure we are inside the matrix verticallly
  if (threadyInd < matrixHeight) {
  
    // go through the threads vertically and sum them into a variable
    for (int i=0; i<blockElt; i++)
      // A col index   : blockIdx.x * BLOCK_WIDTH + i : blockxInd + i
      // A row index  : blockIdx.y * BLOCK_HEIGHT + threadIdx.x : blockyInd + threadIdx.x : threadyInd
      // B index : b[i]

      // cSum = B index * ( A col index * matrixHeight + A row index)
      cSum += b[i] * a[(blockxInd + i) * (matrixHeight) + (threadyInd)];
      //printf("csum = %f\n", cSum);
    
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

  int threads_perblockm = min(n_cols, max_threads_per_block);
  dim3 threadsPerBlockm(threads_perblockm);
  int num_blocksm = (int)ceil((float)n_cols/(float)threads_perblockm);
  dim3 numBlocksm(num_blocksm);

  int blockCols = (int) ceil(n_rows / (double) BLOCK_WIDTH);
  int blockRows = (int) ceil(n_cols / (double) BLOCK_HEIGHT);
  dim3 dimBlock(BLOCK_HEIGHT);
  dim3 dimGrid(blockCols, blockRows);

  int sharedMem = 3 * sizeof (int) + BLOCK_WIDTH * sizeof(_Tp);

  // execute kernels
  zero_vector_float<<<numBlocksm, threadsPerBlockm>>>(C, n_cols);
  MatMulKernel<<<dimGrid, dimBlock, sharedMem>>>(C, B, A, n_cols, n_rows);

  cudaDeviceSynchronize();
   
  // Convert product vector to xtensor
  xt::xarray<double>::shape_type C_shape = {size_C, 1};
  xt::xarray<float> vec_C = xt::adapt(C, size_C, xt::no_ownership(), C_shape);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  
  return vec_C;
}


int main(int argc, char *argv[])
// int main()
{
//     int m = 512;
//     int n = 1024;
  
//     // declare matrices for CPU and allocate memory
//     float *A = (float *) malloc (m * n * sizeof(float));
//     float *B = (float *) malloc (n * sizeof(float));
//     float *C = (float *) malloc (m * sizeof(float));
    
//     // randomly fill in elements of CPU matrices
//     createRandomMatrix(A, m * n, time(NULL));
//     createRandomMatrix(B, n, time(NULL));
    
//     matVecMul (C, B, A, m, n);
    
//     free(A);
//     free(B);
//     free(C); 
  
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
  
  std::cout << "Weight Matrix" << std::endl << tr_dense_weights << std::endl;
  std::cout << "***************************************" << std::endl;
  std::cout << "Input Vector" << std::endl << input_image << std::endl;
  std::cout << "***************************************" << std::endl;
  std::cout << "Output Vector" << std::endl << l1 << std::endl;
  std::cout << "***************************************" << std::endl;


//   // first layer bias
//   xt::xarray<float> b1 = l1 + dense_biases;
    
//   // relu activation
//   xt::xarray<float> b1_relu = relu<float>(b1);

//   /******************LAYER 2******************/
//   // transpose weight matrix
//   xt::xarray<float> tr_dense_1_weights = xt::transpose(dense_1_weights);
    
//   // send through layer
//   xt::xarray<float> l2 = matVecMul<float>(tr_dense_1_weights, b1_relu);
    
//   // second layer bias
//   xt::xarray<float> b2 = l2 + dense_1_biases;
    
//   // softmax activation
//   xt::xarray<float> l3 = softmax(b2);
    
//   // argmax
//   std::cout << "PREDICTION:   " << xt::argmax(l3, 0)[0] << std::endl;
    
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