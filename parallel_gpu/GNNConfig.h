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
	template<typename T>
		__device__ inline T D(T x){
			return F(x)*F(1-x);
		}

		template<typename T>
		__device__ inline T F(T x){
			return 1/(1 + exp(-x));
		}

		template<typename T>
		__device__ inline T operator()(T x){
			return F(x);
		}
};

template<typename DATA_T, typename ACT_F>
struct GNeuralNetwork{


};

template<typename DATA_T, typename ACT_F>
struct Layer{
	DATA_T *M_j;
	DATA_T *v_j;



};



};

#endif
