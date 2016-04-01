#include "CudaHelper.h"

/*
 * Setup kernel for random number generator
 *
 */
__global__ void setup_kernel()
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	/* Each thread gets different seed, a different sequence
	number, no offset */
	curand_init(7 + id, id, 0, &devStates[id]);
}

/*
 * Initializing random State in GPU
 *
 */
__host__ void init_rand_gpu(){
	setup_kernel<<< RAND_BLOCKS, RAND_THREADS >>>();
	error = cudaDeviceSynchronize();
	handleDeviceErrors(error, "Error Initializing Rand GPU in CudaHelper");
}

/*
 * Device error handling
 *
 */
__host__ void handleDeviceErrors(cudaError_t error, std::string comment){
	if (error != cudaSuccess){ std::cout << "Cuda Error: " << comment << "," << cudaGetErrorString(error) << std::endl; }
}

/*
 * Allocating device memory
 * addr: Address in GPU
 * size: Data size in bytes
 * msg: Error message to be displayed
 */
template void allocDevMem<unsigned int>(unsigned int **addr, unsigned int size, std::string msg);
template void allocDevMem<unsigned short>(unsigned short **addr, unsigned int size, std::string msg);
template void allocDevMem<int>(int **addr, unsigned int size, std::string msg);
template void allocDevMem<short>(short **addr, unsigned int size, std::string msg);
template void allocDevMem<bool>(bool **addr, unsigned size, std::string msg);
template void allocDevMem<float>(float **addr, unsigned size, std::string msg);
template void allocDevMem<double>(double **addr, unsigned size, std::string msg);

template<class V>
__host__ void allocDevMem(V **addr, unsigned int size, std::string msg){
	error = cudaMalloc(&(*addr), size);
	handleDeviceErrors(error, "Error Allocating device " + msg);
}

/*
 * Allocating Pinned Memory
 *
 * addr: Address in GPU
 * size: Data size in bytes
 * msg: error message to be displayed
 */
template void allocHostMem<unsigned int>(unsigned int **addr, unsigned int size, std::string msg);
template void allocHostMem<unsigned short>(unsigned short **addr, unsigned int size, std::string msg);
template void allocHostMem<int>(int **addr, unsigned int size, std::string msg);
template void allocHostMem<short>(short **addr, unsigned int size, std::string msg);
template void allocHostMem<bool>(bool **addr, unsigned int size, std::string msg);
template void allocHostMem<float>(float **addr, unsigned int size, std::string msg);
template void allocHostMem<double>(double **addr, unsigned int size, std::string msg);


template<class V>
__host__ void allocHostMem(V **addr, unsigned int size, std::string msg){
	error = cudaMallocHost(&(*addr), size);
	handleDeviceErrors(error, "Error Allocating host "+msg);
}

/*
 * Copy Arrays to Device
 * to: GPU address
 * from: DRAM address
 * size: data size in bytes
 * msg: error message to be displayed
 */

template void safeCpyToDevice<unsigned int>(unsigned int *to, unsigned int *from, unsigned int size, std::string msg);
template void safeCpyToDevice<unsigned short>(unsigned short *to, unsigned short *from, unsigned int size, std::string msg);
template void safeCpyToDevice<int>(int *to, int *from, unsigned int size, std::string msg);
template void safeCpyToDevice<short>(short *to, short *from, unsigned int size, std::string msg);
template void safeCpyToDevice<bool>(bool *to, bool *from, unsigned int size, std::string msg);
template void safeCpyToDevice<float>(float *to, float *from, unsigned int size, std::string msg);
template void safeCpyToDevice<double>(double *to, double *from, unsigned int size, std::string msg);

template<class V>
__host__ void safeCpyToDevice(V *to, V *from, unsigned int size, std::string msg){
	error = cudaMemcpy(to,from,size,cudaMemcpyHostToDevice);
	handleDeviceErrors(error, "Error Copying to device "+ msg);
}

template void safeCpyToHost<unsigned int>(unsigned int *to, unsigned int *from, unsigned int size, std::string msg);
template void safeCpyToHost<unsigned short>(unsigned short *to, unsigned short *from, unsigned int size, std::string msg);
template void safeCpyToHost<int>(int *to, int *from, unsigned int size, std::string msg);
template void safeCpyToHost<short>(short *to, short *from, unsigned int size, std::string msg);
template void safeCpyToHost<bool>(bool *to, bool *from, unsigned int size, std::string msg);
template void safeCpyToHost<float>(float *to, float *from, unsigned int size, std::string msg);
template void safeCpyToHost<double>(double *to, double *from, unsigned int size, std::string msg);

template<class V>
__host__ void safeCpyToHost(V *to, V *from, unsigned int size, std::string msg){
	error = cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost);
	handleDeviceErrors(error, "Error Copying to device " + msg);
}


/*
 * Copying to symbol
 *
 */
template void safeCpyToSymbol<unsigned int>(unsigned int *symbol, unsigned int *data, std::string msg);
template void safeCpyToSymbol<unsigned short>(unsigned short *symbol, unsigned short *data, std::string msg);

template<class V>
__host__ void safeCpyToSymbol(V *symbol, V *data, std::string msg){
	unsigned int k = 13;
	error = cudaMemcpyToSymbol(symbol, &k, sizeof(V));
	handleDeviceErrors(error, "Error Copying symbol "+ msg);
}

/*
 * Print current Device Specs
 *
 */
__host__ cudaError_t printDeviceSpecs(bool print){
	cudaDeviceProp prop;
	cudaError_t error = cudaSuccess;
	int devs = 0;
	
	error = cudaGetDeviceCount(&devs);
	if (!print) return error;
	if (error != cudaSuccess){ handleDeviceErrors(error, "Error Getting Number of Devices");  return error; }
	std::cout << std::endl;
	std::cout << "Number of Devices: (" << devs << ")" << std::endl;

	for (int i = 0; i < devs; i++){
		error = cudaGetDeviceProperties(&prop, i);
		if (error != cudaSuccess){ handleDeviceErrors(error, "Error Reading Device Properties");  return error; }
		std::cout << "<<<<<< Device " << i << " >>>>>>" << std::endl;

		std::cout << "Device Name: " << prop.name << std::endl;

		std::cout << "Device Compute Mode: " << prop.computeMode <<std::endl;
		std::cout << "Device Major Compute Capability: " << prop.major << std::endl;
		std::cout << "Device Minor Compute Capability: " << prop.minor << std::endl;

		std::cout << "Number of AsyncEngineCount: " << prop.asyncEngineCount << std::endl;
		std::cout << "Global Memory Size: " << prop.totalGlobalMem << std::endl;
		std::cout << "Constant Memory Size: " << prop.totalConstMem << std::endl;

		std::cout << "Number of Multiprocessors: " << prop.multiProcessorCount << std::endl;
		std::cout << "Shared Memory Per Multiprocessor: " << prop.sharedMemPerMultiprocessor << std::endl;
		std::cout << "Shared Memory Per Block: " << ((float)prop.sharedMemPerMultiprocessor) << std::endl;
		
		/*int x = 0;
		error = cudaDeviceGetAttribute(&x, cudaDevAttrMaxBlockDimX, 0);
		std::cout << "Device Block Number X:" << x << endl;
		error = cudaDeviceGetAttribute(&x, cudaDevAttrMaxBlockDimY, 0);
		std::cout << "Device Block Number Y:" << x << endl;
		error = cudaDeviceGetAttribute(&x, cudaDevAttrMaxBlockDimZ, 0);
		std::cout << "Device Block Number Z:" << x << endl;*/

		std::cout << "Maximum Grid Size (X,Y,Z): (" << prop.maxGridSize[0] << "),("
			<< prop.maxGridSize[1] << "),(" << prop.maxGridSize[2] << ")" << std::endl;

		std::cout << "Maximum Threads Per Block: " << prop.maxThreadsPerBlock<< std::endl;
		std::cout << "Maximum Number of Blocks (X,Y,Z): (" << prop.maxThreadsDim[0] << "),("
			<< prop.maxThreadsDim[1] << "),(" << prop.maxThreadsDim[2] << ")" << std::endl;

	}
	std::cout << std::endl;

	return cudaSuccess;
}

dim3 grid_1D(unsigned int N, unsigned int data_per_block){
	return dim3((N - 1) / data_per_block + 1, 1, 1);
}

//AMPLIFY = # ELEMENTS PER THREAD
dim3 grid_1D(unsigned int N, unsigned int data_per_block, unsigned int amplification){
	return dim3((N - 1) / (data_per_block*amplification) + 1, 1, 1);
}

dim3 block_1D(unsigned int data_per_block){
	return dim3(data_per_block, 1, 1);
}

void print_grid(dim3 grid, dim3 block){
	std::cout<<"grid("<<grid.x <<","<<grid.y << "," << grid.z <<")"<<std::endl;
	std::cout<<"block("<<block.x <<","<<block.y << "," << block.z <<")"<<std::endl;
}
