#include <cuda_runtime.h>

int main(){

    float* devPtr;
    float *d_a;
    cudaMalloc((void **)&d_a, sizeof(float) * 100);
    cudaFree(devPtr); 
    return 1;
}

// gcc -o test3 test.cpp   -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -ldl -lcudart 
// gdb test3
// set environment LD_PRELOAD=/opt/conda/bin/libevent_hook.so 
// b cudaMalloc
// LD_PRELOAD=/opt/conda/lib/libpython3.so:/opt/conda/bin/libevent_hook.so  ./test3
