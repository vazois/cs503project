#include"common/ArgParser.h"
#include"common/Utils.h"

#include"parallel_gpu/GNNConfig.h"

#include<iostream>

void example_gpu_bench_act(){
	gnn_actf::Sigmoid gs;
	gnn_actf::FSigmoid gfs;
	gnn_actf::Arctan gatan;

	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);
	s.bench_act();

	gnn::GNeuralNetwork<float,gnn_actf::FSigmoid> fs(gfs);
	fs.bench_act();

	gnn::GNeuralNetwork<float,gnn_actf::Arctan> at(gatan);
	at.bench_act();
}

void example_gpu_train(ArgParser ap){
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<double,gnn_actf::Sigmoid> s(gs);
	if(!ap.exists(FIARG)) vz::error("Please provide input file!!!");
	s.loadExamplesFromFile(ap.getString(FIARG));
	std::vector<int> layers;

	layers.push_back(4); //INPUT
	layers.push_back(6); //HIDDEN 1
	layers.push_back(4); // HIDDEN 2
	layers.push_back(8); //HIDDEN 3
	layers.push_back(7); //HIDDEN 4
	layers.push_back(4); //OUTPUT

	s.setBatchSize(4);
	s.createLayers(layers);
	//s.print_weights();
	//s.train();

	//s.bench_test_kernels(MMUL,1112,912,1231,false);
	//s.bench_test_kernels(TMMUL,3,4,3, true);
	//s.bench_test_kernels(TMMUL,618,722,356, false);
	//s.bench_test_kernels(MHPROD,12,5,4, true);
	s.bench_test_kernels(TVECPVEC,7,11,5,true);
}

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	//example_gpu_bench_act(); std::cout<<"<----------------------------------->" << std::endl;
	example_gpu_train(ap);

}
