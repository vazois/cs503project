#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include <string>

#define RAND_BLOCKS 8
#define RAND_THREADS 512
static __device__ curandState devStates[RAND_BLOCKS*RAND_THREADS];

__device__ void rand_num(unsigned int seed);

static cudaError_t error = cudaSuccess;
__host__ void init_rand_gpu();
__host__ cudaError_t printDeviceSpecs(bool);
__host__ void handleDeviceErrors(cudaError_t error, std::string comment);
template<class V> __host__ void allocDevMem(V **addr, unsigned int size, std::string msg);
template<class V> __host__ void allocHostMem(V **addr, unsigned int size, std::string msg);
template<class V> __host__ void safeCpyToSymbol(V *symbol, V *data, std::string msg);
template<class V> __host__ void safeCpyToDevice(V *to, V *from, unsigned int size, std::string msg);
template<class V> __host__ void safeCpyToHost(V *to, V *from, unsigned int size, std::string msg);

dim3 grid_1D(unsigned int N, unsigned int data_per_block);
dim3 grid_1D(unsigned int N, unsigned int data_per_block, unsigned int amplification);
dim3 block_1D(unsigned int data_per_block);
void print_grid(dim3 blocks, dim3 threads);

#endif
