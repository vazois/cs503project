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
	struct LayerBatch{
		unsigned int bsize;
		unsigned int clayer;


		DATA_T *A_j;
		/*
		 * Description:
		 * 		Matrix of activation vectors for layer j.
		 * 		Columns = number of train examples in the batch
		 * 		Rows = number of neurons in the current layer.
		 * Notes:
		 *		First layer matrix size = number of input neurons x batch size
		 */
		LayerBatch(){
		}

		void initLayerBatch(unsigned clz,unsigned int bz){
			bsize = bz;
			clayer = clz;
			allocDevMem(&A_j,sizeof(DATA_T)*bsize*clayer,"Error Allocating Current Layer Batch Matrix");
		}

		~LayerBatch(){
			cudaFree(A_j);
		}
	};

	template<typename DATA_T>
	struct Layer{
		unsigned int clayer;
		unsigned int nlayer;
		DATA_T *W_j = NULL;

		/*
		 * Description:
		 * 		Neural network layer with input weight matrix.
		 * 		Size of matrix is input neurons x output_neurons.
		 */
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

			~GNeuralNetwork(){
				if(network != NULL) delete[] network;
				if(examples != NULL) cudaFreeHost(examples);
				if(batch != NULL) delete[] batch;
			}

			void createLayers(std::vector<int> layers);
			void loadExamplesFromFile(std::string file);
			void train();

			void setBatchSize(unsigned int bz){ this->bsize = bz; }
			void setTransposeExamples(bool transpose){ this->transpose = transpose; }

			/*
			 * Testing methods
			 */
			void bench_act();
			void print_weights();

		private:
			void createLayerBatch();
			void randomInit();

			unsigned int layers = 0;
			unsigned int bsize = 0;//default value.
			bool transpose = true;
			arr2D dimEx;
			LayerBatch<DATA_T> *batch = NULL;
			Layer<DATA_T> *network = NULL;
			DATA_T *examples = NULL;
			DATA_T *batchEx = NULL;
			ACT_F F;
	};

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::loadExamplesFromFile(std::string file){
		IOTools<DATA_T> iot;
		dimEx = iot.dataDim(file);
		//std::cout<<"("<<dim.first<<","<<dim.second<<")"<<std::endl;
		iot.freadFile(examples,file,true);
		//std::cout<< examples[0*dim.first]<<"," << examples[1*dim.first]<<"," << examples[2*dim.first]<<std::endl;
	}

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::createLayers(std::vector<int> layers){
		if(layers.size() <= 0 ) vz::error("Network architecture not valid!");
		this->layers = layers.size();
		network = new Layer<DATA_T>[this->layers-1];

		for(int i = 0;i<this->layers-1;i++){
			network[i].initLayer(layers[i],layers[i+1]);
		}
		randomInit();
	}

	/*
	 * For every batch create multi-layer matrices. Each batch will result in a matrix of activation vectors for each
	 * layer.
	 */
	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::createLayerBatch(){
		if(network == NULL) vz::error("Network architecture missing. Use createLayers first!");
		if(examples == NULL) vz::error("Examples not loaded. Use loadExamplesFromFile!");
		if(bsize > dimEx.second) bsize = dimEx.second;
		if(batch != NULL) delete[] batch;
		if(batchEx != NULL) cudaFree(batchEx);
		batch = new LayerBatch<DATA_T>[this->layers];

		//allocDevMem(&batchEx,sizeof(DATA_T)*network[0].clayer*this->batch,"Error allocating device memory for current batch");
		//safeCpyToDevice<DATA_T>(batchEx,sizeof())
		for(int i = 0; i < this->layers-1;i++){
			batch[i].initLayerBatch(network[i].clayer,this->bsize);
		}
		batch[this->layers - 1].initLayerBatch(network[this->layers-2].nlayer,this->bsize);
	}
}


#endif
