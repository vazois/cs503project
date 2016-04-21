#include "GNNConfig.h"
#include "../common/Time.h"

namespace gnn_kernels{
	template<typename ACT_F>
	__global__ void bench_test_activation(ACT_F F){
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		float a = 0;
		for(int j = 0; j<blockDim.x;j++){
			a+= F.F((float)i*j);
		}
	}

	template<typename DATA_T>
	__global__ void randomWeights(DATA_T *W_j,unsigned int clayer, unsigned int nlayer){
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		if( i < clayer * nlayer){
			W_j[i] = i;
		}
	}

}

namespace gnn{

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::randomInit(){
		std::cout<<"Initializing random weights: "<<std::endl;
		cudaSetDevice(0);

		for(int i = 0;i < layers-1;i++){
			std::cout<<network[i].clayer << "-->" << network[i].nlayer << std::endl;
			unsigned int vector_size = network[i].nlayer * network[i].clayer;
			dim3 grid = grid_1D(vector_size,256);
			dim3 block = block_1D(256);
			gnn_kernels::randomWeights<DATA_T><<<grid,block>>>(network[i].W_j,network[i].clayer,network[i].nlayer);
		}
	}

	/*
	 * Testing methods
	 */

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::bench_act(){
		cudaSetDevice(0);
		dim3 block(512,1,1);
		dim3 grid(128,1,1);

		/*
		 * Warm up device
		 */
		gnn_kernels::bench_test_activation<ACT_F><<<grid,block>>>(this->F);
		cudaDeviceSynchronize();
		/* <END> */

		std::string msg("Benchmark ");
		msg.append(F.TAG);
		Time<millis> t;
		t.start();
		gnn_kernels::bench_test_activation<ACT_F><<<grid,block>>>(F);
		cudaDeviceSynchronize();
		t.lap(msg);
	}

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::print_weights(){
		DATA_T *cW_j;
		cudaSetDevice(0);

		for(int i = 0;i < layers-1;i++){
			unsigned int size = network[i].nlayer * network[i].clayer;
			allocHostMem<DATA_T>(&cW_j,sizeof(DATA_T)*size, "Error Allocating Host Weight Matrix");
			safeCpyToHost<DATA_T>(cW_j,network[i].W_j,sizeof(DATA_T)*size, "Error Allocating Copying Weight Matrix From Device");
		}

	}

	template class GNeuralNetwork<float,gnn::Sigmoid>;
	template class GNeuralNetwork<float,gnn::FSigmoid>;
	template class GNeuralNetwork<float,gnn::Arctan>;
}
