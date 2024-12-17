# Copyright 2024 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -fvisibility=hidden can not hidden those symbols, may be declaration with extern C or visibility sets to default.
# bthread_jump_fcontext
# bthread_make_fcontext
# _ZN6google8protobuf8internal8byteswapILi1EEEvPv
# _ZN6google8protobuf8internal8byteswapILi4EEEvPv
# _ZN6google8protobuf8internal8byteswapILi8EEEvPv
{
    global:
        cublasGemmEx;
        cublasGemmStridedBatchedEx;
        cublasSgemm;
        cublasSgemmStridedBatched;
        cudaLaunchKernel;
        cudaLaunchKernelExC;
        cublasLtMatmul;
        ncclAllReduce;
        ncclReduce;
        ncclAllGather;
        ncclReduceScatter;
        ncclSend;
        ncclRecv;
        ncclBroadcast;
        cudaMalloc;
        cudaMallocFromPoolAsync;
        cudaFreeAsync;
        cudaMemcpyAsync;
        cudaFree;
        cudaHostAlloc;
        cudaMallocHost;
        hggcLaunchKernel;
    local:
        *;
};
