#include"common/ArgParser.h"
#include"common/Utils.h"

#include"serial/SNeuralNet.h"

#include<iostream>

void example_layer_initialization(){
	stdaf::Sigmoid s;//sigmoid activation function
	stdaf::FSigmoid fs;// fast approximate sigmoid function

	Layer<float,stdaf::Sigmoid> ls(2,3,s);//template float values, sigmoid function
	// arguments 2 input neuros , 3 output neurons, sigmoid function definition


	Layer<double,stdaf::FSigmoid> lfs(4,2,fs);//template double values, fast  sigmoid function
		// arguments 4 input neuros , 2 output neurons, fast sigmoid function definition


	std::cout<< "sigmoid activation function test: "<<ls(1.2) << std::endl;
	std::cout<< "fast sigmoid activation function test: "<<lfs(1.2) << std::endl;
}

int main(){
	example_layer_initialization();

	std::vector<Layer<float,stdaf::Sigmoid>> s;

	s.push_back(Layer<float,stdaf::Sigmoid>(2,3,stdaf::Sigmoid()));
	SNeuralNet<float,float,float,stdaf::Sigmoid> nn;

	vz::pause();
}
