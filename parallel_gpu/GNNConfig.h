#ifndef GNNCONFIG_H
#define GNNCONFIG_H

#include<vector>

#include "../common/CudaHelper.h"
#include "../common/IOTools.h"

enum UnitTest{
		MMUL,//MATRIX MULTIPLICATION
		TMMUL,//TRANSPOSE MATRIX MULTIPLICATION
		OUTDM,//OUTPUT DELTA COMPUTATION
		MHPROD,//MATRIX HADAMARD PRODUCT
		TVECPVEC// TRANSPOSE VECTOR PRODUCT VECTOR
	};

#define ZEROS 0
#define ONES 1
#define RANDOM 2

namespace gnn_actf{
	struct Sigmoid{
		char TAG[10] = "Sigmoid";
		template<typename T>
		__forceinline__ __device__ T D(T x){
			return F(x) * (1 - F(x));
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
}

namespace gnn_data{

	template<typename DATA_T>
	struct LayerBatch{
		unsigned int bsize;
		unsigned int clayer;

		DATA_T *A_j;
		DATA_T *D_j;
		DATA_T *Y;
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

		void initLayerBatch(unsigned clz,unsigned int bz, bool input){
			bsize = bz;
			clayer = clz;
			allocDevMem<DATA_T>(&A_j,sizeof(DATA_T)*bsize*clayer,"Error Allocating Activation Layer Batch Matrix");
			if(!input) allocDevMem<DATA_T>(&D_j,sizeof(DATA_T)*bsize*clayer,"Error Allocating Delta Layer Batch Matrix");
		}

		void initOutputBatch(){
			allocDevMem<DATA_T>(&Y, sizeof(DATA_T)*clayer * bsize, "Error allocating Output Y matrix");
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
				allocDevMem<DATA_T>(&W_j, sizeof(DATA_T)*clayer*nlayer, "Error Allocating Weight Matrix");
			}

			~Layer(){
				if(W_j != NULL)cudaFree(W_j);
			}

		};
}

namespace gnn{
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
			void setLearningRate(double lrate){ this->lrate = lrate; }

			/*
			 * Testing methods
			 */
			void bench_act();
			void print_weights();
			void bench_test_kernels(UnitTest test,unsigned int m, unsigned int n, unsigned int k, bool debug);

		private:
			void createLayerBatch();
			void randomInit();

			unsigned int layers = 0;
			unsigned int bsize = 0;//default value.
			double lrate =0.314;
			bool transpose = true;

			arr2D dimEx;
			gnn_data::LayerBatch<DATA_T> *batch = NULL;
			gnn_data::Layer<DATA_T> *network = NULL;
			DATA_T *examples = NULL;
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
		network = new gnn_data::Layer<DATA_T>[this->layers-1];

		/*clayer+1 = Weight matrix includes additional column for bias. nlayer x (clayer + 1)*/
		for(int i = 0;i<this->layers-1;i++){
			network[i].initLayer(layers[i]+1,layers[i+1]);
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
		batch = new gnn_data::LayerBatch<DATA_T>[this->layers];

		/*(clayer - 1) = Activation does not include bias vector*/
		batch[0].initLayerBatch(network[0].clayer-1,this->bsize,true);
		//std::cout<< network[0].clayer-1 <<"<>"<<this->bsize << std::endl;
		/*nlayer is current layer without bias vector for activation matrix*/
		for(int i = 0; i < this->layers-1;i++){
			//std::cout<< network[i].nlayer <<"<>"<< this->bsize << std::endl;
			batch[i+1].initLayerBatch(network[i].nlayer,this->bsize,false);
		}
		batch[this->layers-1].initOutputBatch();//Initialize Y Matrix
	}
}


#endif
