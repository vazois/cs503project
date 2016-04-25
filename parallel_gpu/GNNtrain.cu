#include "GNNConfig.h"
#include "../common/Time.h"

#define LOAD_TILE 4
#define BATCH_TILE 2

namespace gnn_kernels{

	/*
	 * Testing activation functions on kernels.
	 */
	template<typename ACT_F>
	__global__ void bench_test_activation(ACT_F F){
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		float a = 0;
		for(int j = 0; j<blockDim.x;j++){
			a+= F.F((float)i*j);
		}
	}

	/*
	 * Initialize matrices random weights
	 */
	template<typename DATA_T>
	__global__ void randomWeights(DATA_T *W_j,unsigned int clayer, unsigned int nlayer){
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		if( i < clayer * nlayer){
			//W_j[i] = cudaUniRand(i);
			W_j[i] = i;
		}
	}
	/*
	 * Load current batch of train examples.
	 * 		1: First layer batch array.
	 * 		2: Training example matrix.
	 * 		3: Input layer dimension
	 * 		4: Batch size dimension
	 * 		5: Offset indicating the batch being loaded.
	 * 	Notes:
	 * 		Transpose version assumes that the training examples matrix is stored
	 * 		in a row-wise manner.
	 */
	template<typename DATA_T,unsigned int TILE>
	__global__ void loadBatchT(DATA_T *A_j, DATA_T *tEx,
			unsigned int clayer, unsigned int bsize, unsigned int offset){
		__shared__ DATA_T sAj[TILE*TILE];
		int by = blockIdx.y, bx = blockIdx.x;
		int ty = threadIdx.y, tx = threadIdx.x;
		int boffset = (by * clayer + bx ) * TILE;

		sAj[ty*TILE + tx] = tEx[boffset + offset + ty * clayer + tx];
		__syncthreads();
		boffset = (bx * bsize + by ) * TILE;
		A_j[boffset + ty * bsize + tx] = sAj[tx * TILE + ty];
		__syncthreads();
	}

	template<typename DATA_T>
	__global__ void loadBatch(DATA_T *A_j, DATA_T *tEx,
			unsigned int clayer, unsigned int bsize, unsigned int offset){
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int step = gridDim.x * blockDim.x;

		while (i < clayer * bsize){
			A_j[i] = tEx[offset + i];
			i+=step;
		}
	}

	/*
	 * Compute matrix of activation values for a single layer of a given batch.
	 *		1:	Current layer weight matrix.
	 *		2: 	Current layer matrix of activation vectors.
	 *		3: 	Next layer matrix of activation vectors.
	 *		4:	W_j = nlayer x clayer , A_j = clayer x bsize, A_jj = nlayer x bsize.
	 *		5: 	Offset: 0 for hidden and output layer, corresponding row of training example matrix for input layer.
	 */
	template<typename DATA_T, typename ACT_F, unsigned int TILE>
	__global__ void batchActivation(
			DATA_T *A_jj, DATA_T *W_j, DATA_T *A_j, ACT_F F,
			unsigned int clayer, unsigned int nlayer, unsigned int bsize){
		__shared__ DATA_T sWj[TILE * TILE];
		__shared__ DATA_T sAj[TILE * TILE];

		int row = ( blockIdx.y * TILE + threadIdx.y ) * clayer + threadIdx.x;
		int col = ( blockIdx.x * TILE + threadIdx.y * bsize) + threadIdx.x;
		int Ajj = 0;

		for(int i = 0; i < clayer / TILE ; i++){
			sWj[threadIdx.y*TILE + threadIdx.x] = W_j[row];
			sAj[threadIdx.y*TILE + threadIdx.x] = A_j[col];
			__syncthreads();
			for(int j = 0; j < TILE; j++){
				Ajj += sWj[threadIdx.y * TILE + j] * sAj[j * TILE + threadIdx.x];
				//Ajj += sWj[threadIdx.y * TILE + j];
			}
			row += TILE;
			col += bsize * TILE;
			__syncthreads();
		}
		//A_jj[(blockIdx.y * bsize + blockIdx.x) * TILE + threadIdx.y * bsize + threadIdx.x]
			// = (blockIdx.y * bsize + blockIdx.x) * TILE + threadIdx.y * bsize + threadIdx.x;
		//A_jj[(blockIdx.y * bsize + blockIdx.x) * TILE + threadIdx.y * bsize + threadIdx.x] = W_j[row];
		//A_jj[(blockIdx.y * bsize + blockIdx.x) * TILE + threadIdx.y * bsize + threadIdx.x] = A_j[col];
			A_jj[(blockIdx.y * bsize + blockIdx.x) * TILE + threadIdx.y * bsize + threadIdx.x] = Ajj;
	}

	template<typename DATA_T>
	__global__ void printGPU(DATA_T *A, unsigned int row, unsigned int col){
		for(int i =0;i<row*col;i++){
			printf("%.2f ", A[i]);
			if((i+1)% col == 0) printf("\n");
		}
		printf("<-------------------------------------------->\n");
	}

}

namespace gnn{

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::randomInit(){
		if(network == NULL) vz::error("Network architecture missing. Use createLayers first!");
		//std::cout<<"Initializing random weights: "<<std::endl;
		cudaSetDevice(0);

		cudaInitRandStates();
		for(int i = 0;i < layers-1;i++){
			//std::cout<<network[i].clayer << "-->" << network[i].nlayer << std::endl;
			unsigned int vector_size = network[i].nlayer * network[i].clayer;
			dim3 grid = grid_1D(vector_size,256);
			dim3 block = block_1D(256);
			gnn_kernels::randomWeights<DATA_T><<<grid,block>>>(network[i].W_j,network[i].clayer,network[i].nlayer);
		}
	}

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::train(){
		if(network == NULL) vz::error("Network architecture missing. Use createLayers first!");
		if(bsize == 0) vz::error("Batch size not set. Use setBatchSize first!");
		unsigned int nbatch = dimEx.second / this->bsize; //std::cout<< "Batch num: " << nbatch << std::endl;
		createLayerBatch();

		for(int i = 0;i < 1;i++){
			LayerBatch<DATA_T> flayer = batch[0];
			unsigned int bRow = i * this->bsize * flayer.clayer;

			dim3 lgrid((flayer.clayer-1)/LOAD_TILE + 1,(flayer.bsize - 1)/LOAD_TILE + 1,1);
			dim3 lblock(LOAD_TILE,LOAD_TILE,1);
			/*
			 * Load current batch of training examples.
			 */
			if(this->transpose){
				gnn_kernels::loadBatchT<DATA_T,LOAD_TILE><<<lgrid,lblock>>>(flayer.A_j,examples,flayer.clayer,flayer.bsize,bRow);
			}else{
				gnn_kernels::loadBatch<DATA_T><<<32,256>>>(flayer.A_j,examples,flayer.clayer,flayer.bsize,bRow);
			}
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing load batch");

			//std::cout<<flayer.clayer << " x " << flayer.bsize << std::endl;
			//print_grid(lgrid,lblock);
			//gnn_kernels::printGPU<DATA_T><<<1,1>>>(flayer.A_j,flayer.clayer,flayer.bsize);

			for(int j = 0;j < 1;j++){
				//dim3 agrid((lbatch[]));
				dim3 agrid((batch[j+1].bsize - 1)/BATCH_TILE + 1, (batch[j+1].clayer - 1)/BATCH_TILE + 1);
				dim3 ablock(BATCH_TILE,BATCH_TILE);
				gnn_kernels::batchActivation<DATA_T,ACT_F,BATCH_TILE><<<agrid,ablock>>>
						(
								batch[j+1].A_j,
								network[j].W_j,
								batch[j].A_j,
								F,
								network[j].clayer,
								network[j].nlayer,
								batch[j].bsize
						);
				print_grid(agrid,ablock);
				gnn_kernels::printGPU<DATA_T><<<1,1>>>(network[j].W_j,network[j].nlayer,network[j].clayer);
				gnn_kernels::printGPU<DATA_T><<<1,1>>>(batch[j].A_j,batch[j].clayer,batch[j].bsize);
				gnn_kernels::printGPU<DATA_T><<<1,1>>>(batch[j+1].A_j,batch[j+1].clayer,batch[j+1].bsize);
			}
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

		for(int i = 0;i < 1;i++){
			unsigned int size = network[i].nlayer * network[i].clayer;
			allocHostMem<DATA_T>(&cW_j,sizeof(DATA_T)*size, "Error Allocating Host Weight Matrix");
			safeCpyToHost<DATA_T>(cW_j,network[i].W_j,sizeof(DATA_T)*size, "Error Allocating Copying Weight Matrix From Device");

			for(int j = 0;j<size;j++){
				std::cout<<cW_j[j] << " ";
				if((j+1)%network[i].clayer == 0) std::cout<<std::endl;
			}
			std::cout<<std::endl;
		}

	}

	template class GNeuralNetwork<float,gnn::Sigmoid>;
	template class GNeuralNetwork<float,gnn::FSigmoid>;
	template class GNeuralNetwork<float,gnn::Arctan>;
}
