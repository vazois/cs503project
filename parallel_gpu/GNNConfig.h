#ifndef GNNCONFIG_H
#define GNNCONFIG_H

#include "../common/CudaHelper.h"


namespace gnn{
/*
 * GPU thread work organization for feed forward step.
 * - Multiple training examples for a single block.
 * - Activation of neurons for different examples concurrently.
 * - Shared memory used to store weight matrix
 */

	struct Sigmoid{
		char TAG[10] = "Sigmoid";
		template<typename T>
		__forceinline__ __device__ T D(T x){
			return F(x)*F(1-x);
		}

		template<typename T>
		__forceinline__ __device__ T F(T x){
			return 1/(1 + expf(-x));
		}
	};

	struct FSigmoid{
		char TAG[10] = "FSigmoid";
		template<typename T>
		__forceinline__ __device__ T D(T x){
			return 1.0/powf(1.0 + fabsf(x),2.0);
		}

		template<typename T>
		__forceinline__ __device__ T F(T x){
			return x/(1.0 + fabsf(x));
		}
	};

	struct Arctan{
		char TAG[10] = "Arctan";
		template<typename T>
		__forceinline__ __device__ T D(T x){
			return powf(1/acoshf(x),2.0);
		}

		template<typename T>
		__forceinline__ __device__ T F(T x){
			return 	tanhf(x);
		}
	};

	template<typename DATA_T>
	struct TrainExample{

	};

	template<typename DATA_T>
	struct BlockPart{

	};

	template<typename DATA_T, typename ACT_F>
	struct Layer{
		DATA_T *M_j;
		DATA_T *a_j;

		Layer(unsigned int input_nn, unsigned int output_nn, ACT_F F){
			allocDevMem(&M_j,sizeof(DATA_T)*input_nn*output_nn,"error allocating layer weight matrix");
			allocDevMem(&a_j,sizeof(DATA_T)*output_nn,"error allocating layer activation vector");
		}

	};

	template<typename DATA_T, typename ACT_F>
	void bench_act(ACT_F F);

}


#endif
