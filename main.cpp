#include"common/ArgParser.h"
#include"common/Utils.h"

#include"serial/SNeuralNet.h"
#include"parallel_gpu/GNNConfig.h"

#include<iostream>

void example_layer_initialization(){
	stdaf::Sigmoid s;//sigmoid activation function
	stdaf::FSigmoid fs;// fast approximate sigmoid function

	Layer<float,stdaf::Sigmoid> ls(2,3,s);//template float values, sigmoid function
	// arguments 2 input neurons , 3 output neurons, sigmoid function definition


	Layer<double,stdaf::FSigmoid> lfs(4,2,fs);//template double values, fast  sigmoid function
		// arguments 4 input neurons , 2 output neurons, fast sigmoid function definition


	std::cout<< "sigmoid activation function test: "<<ls(1.2) << std::endl;
	std::cout<< "fast sigmoid activation function test: "<<lfs(1.2) << std::endl;

	/*Example neural network construction*/
	std::vector<Layer<float,stdaf::Sigmoid>*> nn;

	nn.push_back(new Layer<float,stdaf::Sigmoid>(2,3,stdaf::Sigmoid()));/*2 inputs 3 nodes in first hidden layer*/
	nn.push_back(new Layer<float,stdaf::Sigmoid>(2,3,stdaf::Sigmoid()));/*3 nodes prev hidden layer, 4 nodes next hidden layer */
	nn.push_back(new Layer<float,stdaf::Sigmoid>(4,1,stdaf::Sigmoid()));/*4 nodes prev hidden layer, 1 output node*/

	/*Testing activation function*/
	std::cout<< "First layer: " << nn[0]->F(1.2) <<std::endl;
	std::cout<< "Second layer: " << nn[1]->F(3.1) << std::endl;
	std::cout<< "Output layer: " << nn[2]->F(0.5) << std::endl;
}

void example_gpu_bench_act(){
	gnn::Sigmoid gs;
	gnn::FSigmoid gfs;
	gnn::Arctan gatan;
	gnn::bench_act<float,gnn::Sigmoid>(gs);
	gnn::bench_act<float,gnn::FSigmoid>(gfs);
	gnn::bench_act<float,gnn::Arctan>(gatan);
}

int main(){
	example_layer_initialization();
	example_gpu_bench_act();

}
