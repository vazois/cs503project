#include "GNNConfig.h"
#include "../common/Time.h"

namespace gnn{

	template<typename ACT_F>
	__global__ void bench_test_activation(ACT_F F){
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		float a = 0;
		for(int j = 0; j<blockDim.x;j++){
			a+= F.F((float)i*j);
		}
	}

	template<typename DATA_T, typename ACT_F>
	__global__ void ffactivation_markI(DATA_T* M, DATA_T* a_j, DATA_T a_jj, ACT_F F){

	}

	template<typename DATA_T, typename ACT_F>
	void bench_act(ACT_F F){
		cudaSetDevice(0);
		dim3 block(512,1,1);
		dim3 grid(128,1,1);

		/*
		 * Warm up device
		 */
		bench_test_activation<ACT_F><<<grid,block>>>(F);
		cudaDeviceSynchronize();
		/* <END> */

		std::string msg("Benchmark ");
		msg.append(F.TAG);
		Time<millis> t;
		t.start();
		bench_test_activation<ACT_F><<<grid,block>>>(F);
		cudaDeviceSynchronize();
		t.lap(msg);
	}



	/*
	 * Template initialization
	 */
	template void bench_act<float,gnn::Sigmoid>(gnn::Sigmoid F);
	template void bench_act<float,gnn::FSigmoid>(gnn::FSigmoid F);
	template void bench_act<float,gnn::Arctan>(gnn::Arctan F);
}
