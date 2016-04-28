#include "GNNConfig.h"
#include "../common/Time.h"

#define LOAD_TILE 4
#define ACT_TILE 32
#define DELTA_TILE 16

#define DPT 4 //DATA PER THREADS
#define BSIZE 512

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
			//W_j[i] = i/1024 + 128;
			W_j[i] = 0.1 * i;
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
	__global__ void	mmul(
			DATA_T *A_jj,
			DATA_T *W_j,
			DATA_T *A_j,
			ACT_F F,
			unsigned int nlayer,
			unsigned int clayer,
			unsigned int bsize
			)
	{
		__shared__ DATA_T sWj[TILE * TILE];
		__shared__ DATA_T sAj[TILE * TILE];

		int row = ( blockIdx.y * blockDim.y + threadIdx.y );
		int col = ( blockIdx.x * blockDim.x + threadIdx.x );
		DATA_T Ajj = 0;

		int loadOffset = threadIdx.y*TILE + threadIdx.x;
		for(int i = 0;i < ((clayer - 1) / TILE) + 1; i++){
			if( row < nlayer && (i * TILE + threadIdx.x ) < clayer)
				sWj[loadOffset] = W_j[ row * clayer + i * TILE  + threadIdx.x];
			else sWj[loadOffset] = 0.0;

			if ( i*TILE + threadIdx.y < clayer && col < bsize )
				sAj[loadOffset] = A_j[(i * TILE + threadIdx.y) * bsize + col];
			else sAj[loadOffset] = 0.0;
			__syncthreads();

			for(int j = 0;j < TILE; j++) Ajj += sWj[threadIdx.y * TILE + j] * sAj[j * TILE + threadIdx.x];
			__syncthreads();
		}
		if( row < nlayer && col < bsize ) A_jj[row * bsize + col ] = Ajj;//TODO: enable activation//
	}

	/*
	 * Kernel that computes the last layer difference between the batch activation matrix and the expected output
	 * matrix.
	 */
	template<typename DATA_T>
	__global__ void outputD(
			DATA_T *D_j,
			DATA_T *ExA_j,
			DATA_T *A_j,
			unsigned int size
		)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if ( i < size){ D_j[i] = ExA_j[i] - A_j[i]; }
	}

	/*
	 * Transpose matrix multiplication.
	 *  D_j = (W_j)^T . D_jj
	 */
	template<typename DATA_T, unsigned int TILE>
	__global__ void tmmul(
			DATA_T *D_j,
			DATA_T *W_j,
			DATA_T *D_jj,
			unsigned int clayer,
			unsigned int nlayer,
			unsigned int bsize
			)
	{
		__shared__ DATA_T sWj[TILE * TILE];
		__shared__ DATA_T sDjj[TILE * TILE];

		DATA_T Dj = 0.0;
		int colW = ( blockIdx.y * blockDim.y + threadIdx.x );// by * TILE + ty * clayer + threadIdx.x
		int colD = ( blockIdx.x * blockDim.x + threadIdx.x );

		int loadOffset = threadIdx.y*TILE + threadIdx.x;
		for(int i = 0; i < (nlayer - 1) / TILE + 1 ; i++){
			if( (i * TILE +  threadIdx.y) < nlayer && colW < clayer)
				sWj[loadOffset] = W_j[ (i * TILE +  threadIdx.y) * clayer + colW ];
			else
				sWj[loadOffset] = 0.0;

			if((i * TILE + threadIdx.y) < nlayer && colD < bsize)
				sDjj[loadOffset] = D_jj[ (i * TILE + threadIdx.y) * bsize + colD ];
			else
				sDjj[loadOffset] = 0.0;
			__syncthreads();

			for(int j=0;j<TILE;j++) Dj += sWj[j * TILE + threadIdx.y] * sDjj[j * TILE + threadIdx.x];
			__syncthreads();
		}

		int row = ( blockIdx.y * blockDim.y + threadIdx.y );
		if( row < clayer && colD < bsize) D_j[row * bsize + colD] = Dj;
	}

	/*
	 * Matrix Hadamard product
	 */
	template<typename DATA_T, typename ACT_F>
	__global__ void mhprod(
			DATA_T *D_j,
			DATA_T *A_j,
			ACT_F F,
			unsigned int clayer,
			unsigned int bsize
		){

		int step = gridDim.x * blockDim.x;
		int i = blockIdx.x * blockDim.x + threadIdx.x;

		while( i < clayer * bsize){
			DATA_T Aj = A_j[i];
			D_j[i] *= F.D(Aj);
			i+=step;
		}
	}

	template<typename DATA_T>
	__global__ void printGPU(DATA_T *A, unsigned int row, unsigned int col){
		for(int i =0;i<row*col;i++){
			printf("%.6f ", A[i]);
			if((i+1)% col == 0) printf("\n");
		}
		//printf("<-------------------------------------------->\n");
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
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing randomWeights kernel");
		}
	}

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::train(){
		if(network == NULL) vz::error("Network architecture missing. Use createLayers first!");
		if(bsize == 0) vz::error("Batch size not set. Use setBatchSize first!");
		unsigned int nbatch = dimEx.second / this->bsize; //std::cout<< "Batch num: " << nbatch << std::endl;
		createLayerBatch();

		for(int i = 0;i < 1;i++){
			gnn_data::LayerBatch<DATA_T> flayer = batch[0];
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

			/*
			 * Neural network feed forward step.
			 */
			for(int j = 0;j < this->layers - 1;j++){
				dim3 agrid((batch[j+1].bsize - 1)/ACT_TILE + 1, (batch[j+1].clayer - 1)/ACT_TILE + 1);
				dim3 ablock(ACT_TILE,ACT_TILE);
				gnn_kernels::mmul<DATA_T,ACT_F,ACT_TILE><<<agrid,ablock>>>
						(
								batch[j+1].A_j,
								network[j].W_j,
								batch[j].A_j,
								F,
								network[j].nlayer,
								network[j].clayer,
								batch[j].bsize
						);
				handleDeviceErrors(cudaDeviceSynchronize(),"Error executing batch activation");

				/*printf(">>>>>>ACTIVATION<<<<<<< %d\n",j);
				printf("B(%d) = W(%d) * B(%d)\n",j+1,j,j);
				print_grid(agrid,ablock);
				gnn_kernels::printGPU<DATA_T><<<1,1>>>(batch[j+1].A_j,batch[j+1].clayer,batch[j+1].bsize);
				cudaDeviceSynchronize();
				gnn_kernels::printGPU<DATA_T><<<1,1>>>(network[j].W_j,network[j].nlayer,network[j].clayer);
				cudaDeviceSynchronize();
				gnn_kernels::printGPU<DATA_T><<<1,1>>>(batch[j].A_j,batch[j].clayer,batch[j].bsize);
				cudaDeviceSynchronize();*/
			}

			/*
			 * Back propagation step.
			 */
			dim3 ogrid = grid_1D(batch[layers-1].clayer * batch[layers-1].bsize, 256);
			dim3 oblock = block_1D(256);
			gnn_kernels::outputD<DATA_T><<<ogrid,oblock>>>(
					batch[layers-1].D_j,
					batch[0].A_j,////TODO: Initialize Y matrix correctly
					batch[layers-1].A_j,
					batch[layers-1].clayer * batch[layers-1].bsize
				);
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing outputD kernel");

			/*printf(">>>>>>Output Delta<<<<<<<\n");
			print_grid(ogrid,oblock);
			gnn_kernels::printGPU<DATA_T><<<1,1>>>(batch[layers-1].D_j,batch[layers-1].clayer,batch[layers-1].bsize);
			cudaDeviceSynchronize();
			gnn_kernels::printGPU<DATA_T><<<1,1>>>(batch[0].A_j,batch[0].clayer,batch[0].bsize);
			cudaDeviceSynchronize();
			gnn_kernels::printGPU<DATA_T><<<1,1>>>(batch[layers-1].A_j,batch[layers-1].clayer,batch[layers-1].bsize);
			cudaDeviceSynchronize();*/

			/*
			 * Backpropagation transpose matrix multiplication.
			 * 		for i = layers-1 : > 1 : i--
			 * 		batch[i-1].D_j = network[i-1].W_j * batch[i].D_j
			 * 		grid = (batch[i-1].bsize / TILE + 1), batch[i-1].clayer / TILE + 1
			 * 		block = (TILE, TILE)
			 */
			for(int j = layers-1; j > 1 ; j--){
					dim3 dgrid((batch[j-1].bsize - 1) / DELTA_TILE + 1, (batch[j-1].clayer - 1) / DELTA_TILE + 1);
					dim3 dblock(DELTA_TILE, DELTA_TILE);


					gnn_kernels::tmmul<DATA_T,DELTA_TILE><<<dgrid,dblock>>>(
							batch[j-1].D_j,//(clayer x bsize)
							network[j-1].W_j,//(nlayer x clayer)
							batch[j].D_j,// (nlayer x bsize)
							network[j-1].clayer,
							network[j-1].nlayer,
							batch[j].bsize
							);
					handleDeviceErrors(cudaDeviceSynchronize(),"Error executing tmmul kernel");

					printf(">>>>>>Hidden Layer Delta<<<<<<<\n");
					//printf("(%d,%d,%d)\n",j-1,batch[j-1].clayer,batch[j-1].bsize);
					//printf("(%d,%d,%d)\n",j-1,network[j-1].nlayer,network[j-1].clayer);
					print_grid(dgrid,dblock);
					gnn_kernels::printGPU<DATA_T><<<1,1>>>(batch[j-1].D_j,batch[j-1].clayer,batch[j-1].bsize);
					cudaDeviceSynchronize();
					gnn_kernels::printGPU<DATA_T><<<1,1>>>(network[j-1].W_j,network[j-1].nlayer,network[j-1].clayer);
					cudaDeviceSynchronize();
					gnn_kernels::printGPU<DATA_T><<<1,1>>>(batch[j].D_j,batch[j].clayer,batch[j].bsize);
					cudaDeviceSynchronize();

					break;
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

	template<typename DATA_T, typename ACT_F>
	void GNeuralNetwork<DATA_T,ACT_F>::bench_test_kernels(UnitTest test,unsigned int m, unsigned int n, unsigned int k,
			bool debug){
		DATA_T *hostA, *hostB, *hostC, *hostD;
		DATA_T *devA, *devB, *devC;

		allocDevMem<DATA_T>(&devA,sizeof(DATA_T) * m * n, "Error allocating devA memory");
		allocDevMem<DATA_T>(&devB,sizeof(DATA_T) * n * k, "Error allocating devB memory");
		allocDevMem<DATA_T>(&devC,sizeof(DATA_T) * m * k, "Error allocating devC memory");

		allocHostMem<DATA_T>(&hostA,sizeof(DATA_T) * m * n, "Error allocating devA memory");
		allocHostMem<DATA_T>(&hostB,sizeof(DATA_T) * n * k, "Error allocating devB memory");
		allocHostMem<DATA_T>(&hostC,sizeof(DATA_T) * m * k, "Error allocating devC memory");

		dim3 rgrid;
		dim3 rblock = block_1D(256);
		rgrid = grid_1D(m * n,256); gnn_kernels::randomWeights<DATA_T><<<rgrid,rblock>>>(devA,m,n);
		rgrid = grid_1D(n * k,256); gnn_kernels::randomWeights<DATA_T><<<rgrid,rblock>>>(devB,n,k);
		rgrid = grid_1D(m * k,256); gnn_kernels::randomWeights<DATA_T><<<rgrid,rblock>>>(devC,m,k);

		if(test == MMUL){
			dim3 agrid((k - 1)/ACT_TILE + 1, (m - 1)/ACT_TILE + 1);
			dim3 ablock(ACT_TILE,ACT_TILE);
			Time<millis> t;
			t.start();
			gnn_kernels::mmul<DATA_T,ACT_F,ACT_TILE><<<agrid,ablock>>>
					(
							devC,
							devA,
							devB,
							F,
							m,
							n,
							k
					);
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing batch mmul");
			if(!debug) t.lap("GPU mmul elapsed time");

			allocHostMem<DATA_T>(&hostD,sizeof(DATA_T) * m * k, "Error allocating devC memory");
			safeCpyToHost<DATA_T>(hostA,devA,sizeof(DATA_T)*m*n,"Error copying devA to host");
			safeCpyToHost<DATA_T>(hostB,devB,sizeof(DATA_T)*n*k,"Error copying devB to host");
			safeCpyToHost<DATA_T>(hostC,devC,sizeof(DATA_T)*m*k,"Error copying devC to host");

			t.start();
			for(int x = 0; x < m; x++){//3
				for (int y = 0; y < k; y++){//3
					hostD[x * k + y] = 0.0;
					for (int z = 0; z < n; z++){//2
						hostD[x * k + y] += hostA[x * n + z] * hostB[z * k + y];
					}
				}
			}
			if(!debug) t.lap("CPU serial mmul elapsed time");
			if(debug){
				gnn_kernels::printGPU<DATA_T><<<1,1>>>(devC,m,k);
				cudaDeviceSynchronize(); printf("<----->\n");
				/*gnn_kernels::printGPU<DATA_T><<<1,1>>>(devA,m,n);
				cudaDeviceSynchronize(); printf("<----->\n");
				gnn_kernels::printGPU<DATA_T><<<1,1>>>(devB,n,k);
				cudaDeviceSynchronize(); printf("<----->\n");*/
				for(int x = 0; x<m * k;x++){
					printf("%.6f ", hostD[x]);
					if((x+1)%k==0) printf("\n");
				}
			}else{
				for(int x = 0; x<m * k;x++){
					if(((hostD[x] - hostC[x]) > 0.001 )){
						printf("Result matrices do not match(%f,%f)!!!\n",hostD[x],hostC[x] );
					}
				}
			}
			cudaFreeHost(hostD);
		}else if(test == TMMUL){
			// devB = devA * devC
			// (n x k) = (m x n) (m x k) <=> (n x k) = (m x n)^T (m x k) <=> (n x k) = (n x m) (m x k)
			Time<millis> t;
			dim3 agrid((k - 1)/DELTA_TILE + 1, (n - 1)/DELTA_TILE + 1);
			dim3 ablock(DELTA_TILE,DELTA_TILE);
			t.start();
			gnn_kernels::tmmul<DATA_T,DELTA_TILE><<<agrid,ablock>>>(
					devB,//n
					devA,//
					devC,//
					n,
					m,
					k
			);
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing tmmul kernel");
			if(!debug) t.lap("GPU tmmul elapsed time");

			allocHostMem<DATA_T>(&hostD,sizeof(DATA_T) * n * k, "Error allocating devC memory");
			safeCpyToHost<DATA_T>(hostA,devA,sizeof(DATA_T)*m*n,"Error copying devA to host");
			safeCpyToHost<DATA_T>(hostB,devB,sizeof(DATA_T)*n*k,"Error copying devB to host");
			safeCpyToHost<DATA_T>(hostC,devC,sizeof(DATA_T)*m*k,"Error copying devC to host");

			t.start();
			for(int x = 0; x < n; x++){//3
				for (int y = 0; y < k; y++){//3
					hostD[x * k + y] = 0.0;
					for (int z = 0; z < m; z++){//2
						hostD[x * k + y] += hostA[z * n + x] * hostC[z * k + y];
					}
				}
			}
			if(!debug) t.lap("CPU serial mmul elapsed time");

			if(debug){
				print_grid(agrid,ablock);
				gnn_kernels::printGPU<DATA_T><<<1,1>>>(devB,n,k);
				cudaDeviceSynchronize(); printf("<--()-->\n");
				//gnn_kernels::printGPU<DATA_T><<<1,1>>>(devA,m,n);
				//cudaDeviceSynchronize(); printf("<----->\n");
				//gnn_kernels::printGPU<DATA_T><<<1,1>>>(devC,m,k);
				//cudaDeviceSynchronize(); printf("<----->\n");
				for(int x = 0; x<n * k;x++){
					printf("%.6f ", hostD[x]);
					if((x+1)%k==0) printf("\n");
				}
			}else{
				for(int x = 0; x<m * k;x++){
					if(((hostD[x] - hostB[x]) > 0.001 )){
						printf("Result matrices do not match(%f,%f)!!!\n",hostD[x],hostB[x] );
					}
				}
			}
			cudaFreeHost(hostD);
		}else if (test == MHPROD){
			dim3 grid = grid_1D(m * n, BSIZE * DPT);
			dim3 block = block_1D(BSIZE);
			print_grid(grid,block);

			DATA_T *devD;
			allocDevMem<DATA_T>(&devD,sizeof(DATA_T)*m*n,"Error allocating devD memory");
			rgrid = grid_1D(m * n,256); gnn_kernels::randomWeights<DATA_T><<<rgrid,rblock>>>(devD,m,n);

			if(debug){
				gnn_kernels::printGPU<DATA_T><<<1,1>>>(devA,m,n);
				cudaDeviceSynchronize(); printf("<----->\n");
				gnn_kernels::printGPU<DATA_T><<<1,1>>>(devD,m,n);
				cudaDeviceSynchronize(); printf("<----->\n");
			}

			Time<millis> t;
			if(!debug) t.start();
			gnn_kernels::mhprod<DATA_T,ACT_F><<<grid,block>>>(devA,devD,F,n,m);
			handleDeviceErrors(cudaDeviceSynchronize(),"Error executing mhprod kernel");
			if(!debug) t.lap("mhprod elapsed time");

			if(debug){
				gnn_kernels::printGPU<DATA_T><<<1,1>>>(devA,m,n);
				cudaDeviceSynchronize(); printf("<----->\n");
			}
			cudaFree(devD);
		}

		cudaFree(devA); cudaFree(devB); cudaFree(devC);
		cudaFreeHost(hostA); cudaFreeHost(hostB); cudaFreeHost(hostC); cudaFreeHost(hostD);
	}

	template class GNeuralNetwork<float,gnn_actf::Sigmoid>;
	template class GNeuralNetwork<float,gnn_actf::FSigmoid>;
	template class GNeuralNetwork<float,gnn_actf::Arctan>;

	template class GNeuralNetwork<double,gnn_actf::Sigmoid>;
	template class GNeuralNetwork<double,gnn_actf::FSigmoid>;
	template class GNeuralNetwork<double,gnn_actf::Arctan>;
}
