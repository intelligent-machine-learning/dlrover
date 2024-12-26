// Copyright 2024 The DLRover Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

void cudaMallocWrapper(void** devPtr, size_t size) {
  cudaError_t err = cudaMalloc(devPtr, size);

  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  printf("Success");
}

int main() {
  int device = 0;
  cudaSetDevice(device);

  float* devPtr1;
  cudaMallocWrapper((void**)&devPtr1, sizeof(float) * 100);

  cudaMemPoolProps poolProps = {};
  poolProps.allocType = cudaMemAllocationTypePinned;
  poolProps.location.id = device;
  poolProps.location.type = cudaMemLocationTypeDevice;

  cudaMemPool_t pool;
  cudaError_t err = cudaMemPoolCreate(&pool, &poolProps);

  if (err != cudaSuccess) {
    std::cerr << "Failed to create CUDA memory pool: "
              << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  void* devPtr = NULL;
  size_t size = 1024 * sizeof(int);
  cudaError_t cudaStatus = cudaMallocFromPoolAsync(&devPtr, size, pool, 0);
  if (cudaStatus != cudaSuccess) {
    std::cerr << "cudaMallocFromPoolAsync failed: "
              << cudaGetErrorString(cudaStatus) << std::endl;
    return 1;
  }

  int* hostPtr = new int[1024];
  cudaMemcpy(hostPtr, devPtr, size, cudaMemcpyDeviceToHost);

  cudaFree(devPtr);
  cudaMemPoolDestroy(pool);
  delete[] hostPtr;

  return 1;
}
