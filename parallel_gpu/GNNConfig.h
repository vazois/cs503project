#ifndef GNNCONFIG_H
#define GNNCONFIG_H

#include<vector>

#include "../common/CudaHelper.h"
#include "../common/IOTools.h"


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
	struct ForwardBatch{
		unsigned int batch;
		unsigned int clayer;
		unsigned int nlayer;

		DATA_T *A_j;

		ForwardBatch(unsigned int bz, unsigned int clz, unsigned int nlz){
			batch = bz;
			clayer = clz;
			nlayer = nlz;

			//allocDevMem(&W_j, sizeof(DATA_T)*clayer_act_size*nlayer_act_size, "Error Allocating");
			allocDevMem(&A_j,sizeof(DATA_T)*batch*clayer,"Error Allocating Current Layer Batch Matrix");
			//allocDevMem(&A_jj,sizeof(DATA_T)*batch_size*nlayer_act_size,"Error Allocating Next Layer Batch Matrix");
		}

		~ForwardBatch(){
			cudaFree(A_j);
		}
	};

	template<typename DATA_T>
	struct Layer{
		unsigned int clayer;
		unsigned int nlayer;

		DATA_T *W_j = NULL;
		Layer(){

		}

		void initLayer(unsigned int clz, unsigned int nlz){
			clayer = clz;
			nlayer = nlz;
			allocDevMem(&W_j, sizeof(DATA_T)*clayer*nlayer, "Error Allocating Weight Matrix");
		}

		~Layer(){
			if(W_j != NULL)cudaFree(W_j);
		}

	};


	template<typename DATA_T, typename ACT_F>
	class GNeuralNetwork{
		public:
			GNeuralNetwork(ACT_F F){
				this->F = F;
			};

			void createLayers(std::vector<int> layers);
			void loadExamplesFromFile(std::string file);

			/*
			 * Testing methods
			 */
			void bench_act();
			void print_weights();

		private:
			void randomInit();

			unsigned int layers = 0;
			Layer<DATA_T> *network;
			DATA_T* examples;
			ACT_F F;
	};

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::loadExamplesFromFile(std::string file){
		IOTools<DATA_T> iot;
		arr2D dim = iot.dataDim(file);
		//std::cout<<"("<<dim.first<<","<<dim.second<<")"<<std::endl;
		iot.freadFile(examples,file,true);
		//std::cout<< examples[0*dim.first]<<"," << examples[1*dim.first]<<"," << examples[2*dim.first]<<std::endl;
	}

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::createLayers(std::vector<int> layers){
		network = new Layer<DATA_T>[layers.size()-1];

		this->layers = layers.size();
		for(int i = 0;i<layers.size()-1;i++){
			network[i].initLayer(layers[i],layers[i+1]);
		}

		randomInit();
	}



}


#endif
