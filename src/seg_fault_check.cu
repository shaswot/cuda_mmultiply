#include <iostream>
#include <cstdio>

using std::cout;
using std::endl;

__global__
void print_a(int* a, const int n) {
  for (int i=0; i<n; i++) {
    printf("a[%d] = %d\n",i,a[i]);
  }
}

int main() {

  // allocate a
  const int lena = 6;
  int* a;
  cudaMallocManaged(&a,lena*sizeof(*a));

  // allocate b
  const int lenb = 1;
  int *b;
  cudaMallocManaged(&b,lenb*sizeof(*b));

  // set a
  for (int i=0; i<lena; i++) {
    a[i] = i;
  }

  // set b
  *b = 55;

  // print a from device
  print_a<<<1,1>>>(a,lena);

  // add this device sync to fix seg fault
  //cudaDeviceSynchronize();

  // print b from host
  cout << "*b: " << *b << endl;

  cudaFree(a);
  cudaFree(b);
  return 0;
}
