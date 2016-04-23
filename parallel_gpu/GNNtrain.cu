#include "GNNConfig.h"
#include "../common/Time.h"

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
			W_j[i] = cudaUniRand(i);
			//W_j[i] =i;
		}
	}
	/*
	 * Initialize processing batch
	 */
	template<typename DATA_T,unsigned int TILE>
	__global__ void loadTranspose(DATA_T *A_j, DATA_T *tEx,
			unsigned int clayer, unsigned int bsize, unsigned int offset){
		__shared__ DATA_T sAj[TILE*TILE];
		int by = blockIdx.y, bx = blockIdx.x;
		int ty = threadIdx.y, tx = threadIdx.x;
		int boffset = (by * clayer + bx ) * TILE;

		sAj[ty*TILE + tx] = tEx[boffset + offset + ty * clayer + tx];
		__syncthreads();
		boffset = (bx * bsize + by ) * TILE;
		A_j[boffset + ty * bsize + tx] = sAj[tx * TILE + ty];

		if( tx == 0  && ty == 0 && bx == 0 && by == 0){
			//printf("%d>\n",tx);
			for(int  k= 0;k <TILE * TILE;k++){
			//	printf("%f ",sAj[k]);
			//	if(((k+1) % TILE) == 0) printf("\n");
			}
		//	printf("<<>>><\n",tx);
		}
		__syncthreads();
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
			DATA_T *W_j, DATA_T *A_j, DATA_T *A_jj, ACT_F F,
			unsigned int clayer, unsigned int nlayer, unsigned int bsize){
		__shared__ DATA_T sWj[TILE][TILE];
		__shared__ DATA_T sAj[TILE][TILE];

		int by = blockIdx.y;
		int bx = blockIdx.x;

		int ty = threadIdx.y;
		int tx = threadIdx.x;

		int row = by * blockDim.y + ty;
		int col = bx * blockDim.x + tx;
		DATA_T Ajj = 0;

		for(int t = 0; t < TILE ; t++){
			sWj[ty][tx] = W_j[row*clayer + t*TILE + tx];
			sAj[ty][tx] = A_j[(t*TILE + ty)*bsize + col];
			__syncthreads();

			for(int i = 0; i<TILE; i++){
				Ajj += sWj[ty][i] * sAj[i][tx];
			}
			__syncthreads();
		}
		Ajj[row*bsize + col] = F.F(Ajj);
	}

	template<typename DATA_T>
	__global__ void printGPU(DATA_T *A, unsigned int row, unsigned int col){
		for(int i =0;i<row*col;i++){
			printf("%f ", A[i]);
			if((i+1)% col == 0) printf("\n");
		}

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

		std::cout<< "Start Training"<< std::endl;
		unsigned int nbatch = dimEx.second / this->bsize; std::cout<< "Batch num: " << nbatch << std::endl;
		createLayerBatch();

		//for(int i = 0;i < nbatch;i++){
		for(int i = 0;i < 1;i++){
			LayerBatch<DATA_T> flayer = lbatch[0];
			std::cout<<flayer.clayer << " x " << flayer.bsize << std::endl;

			unsigned int TILE = 4;
			unsigned int bRow = i * this->bsize * flayer.clayer ;

			dim3 grid((flayer.clayer-1)/TILE + 1,(flayer.bsize - 1)/TILE + 1,1);
			dim3 block(TILE,TILE,1);
			print_grid(grid,block);
			/*
			 * Load transpose of train examples. Index should be zero
			 */
			if(TILE == 2)
				gnn_kernels::loadTranspose<DATA_T,2><<<grid,block>>>
				(flayer.A_j,examples,flayer.clayer,flayer.bsize,bRow);
			else if(TILE == 4)
				gnn_kernels::loadTranspose<DATA_T,4><<<grid,block>>>
				(flayer.A_j,examples,flayer.clayer,flayer.bsize,bRow);
			else if(TILE == 8)
				gnn_kernels::loadTranspose<DATA_T,8><<<grid,block>>>
				(flayer.A_j,examples,flayer.clayer,flayer.bsize,bRow);
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing load transpose");

			//printDevData<DATA_T>(examples,lbatch[i].bsize,lbatch[i].clayer);
			//printDevData<DATA_T>(flayer.A_j,flayer.bsize,flayer.clayer);
			gnn_kernels::printGPU<DATA_T><<<1,1>>>(flayer.A_j,flayer.clayer,flayer.bsize);
		}


		/*gnn_kernels::batchActivation<DATA_T,ACT_F><<<grid,block>>>(
				network[l].W_j,
				network[l].W_j,
				lbatch[l].A_j,
				F,
				network[l].clayer,
				network[l].nlayer,
				lbatch[l].bsize,
				0
		);*/

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
