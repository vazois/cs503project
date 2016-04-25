#include"common/ArgParser.h"
#include"common/Utils.h"

#include"parallel_gpu/GNNConfig.h"

#include<iostream>

void example_gpu_bench_act(){
	gnn::Sigmoid gs;
	gnn::FSigmoid gfs;
	gnn::Arctan gatan;

	gnn::GNeuralNetwork<float,gnn::Sigmoid> s(gs);
	s.bench_act();

	gnn::GNeuralNetwork<float,gnn::FSigmoid> fs(gfs);
	fs.bench_act();

	gnn::GNeuralNetwork<float,gnn::Arctan> at(gatan);
	at.bench_act();
}

void example_gpu_initializing_weights(ArgParser ap){
	gnn::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn::Sigmoid> s(gs);
	if(!ap.exists(FIARG)) vz::error("Please provide input file!!!");
	s.loadExamplesFromFile(ap.getString(FIARG));
	std::vector<int> layers;

	layers.push_back(8); //INPUT
	layers.push_back(16); //HIDDEN 1
	layers.push_back(12); //HIDDEN 2
	layers.push_back(14); //HIDDEN 3
	layers.push_back(4); //OUTPUT

	s.setBatchSize(4);
	s.createLayers(layers);
	//s.print_weights();

	s.train();
}

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	example_gpu_bench_act(); std::cout<<"<----------------------------------->" << std::endl;
	example_gpu_initializing_weights(ap);

}
