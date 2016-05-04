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

	layers.push_back(8); //INPUT
	layers.push_back(1024); //HIDDEN 1
	layers.push_back(1024); // HIDDEN 2
	layers.push_back(1024); //HIDDEN 3
	layers.push_back(1024); //HIDDEN 4
	layers.push_back(1024); //HIDDEN 5
	layers.push_back(1024); //HIDDEN 6
	layers.push_back(1024); //HIDDEN 7
	layers.push_back(1024); //HIDDEN 8
	layers.push_back(1024); //HIDDEN 9
	layers.push_back(1024); //HIDDEN 10
	layers.push_back(1); //OUTPUT

	s.setBatchSize(1024);
	s.createLayers(layers);
	s.useTranspose(true);
	Time<millis> t;
	t.start();
	s.train();
	t.lap("Training Execution Time");

	//784 100 10
	//
	//s.bench_test_kernels(MMUL,1112,912,1231,false);
	//s.bench_test_kernels(TMMUL,1112,912,1231, false);
	//s.bench_test_kernels(TMMUL,3,4,3, true);
	//s.bench_test_kernels(TMMUL,618,722,356, false);
	//s.bench_test_kernels(MHPROD,3,5,4, true);
	//s.bench_test_kernels(TVECPVEC,457,632,710,false);
	//s.bench_test_kernels(MMUL,3,4,2,true);

	//s.bench_test_kernels(MMUL,3,5,7,true);
	//s.bench_test_kernels(TMMUL,3,5,4,true);
	//s.bench_test_kernels(MHPROD,3,5,4, true);
}

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	//example_gpu_bench_act(); std::cout<<"<----------------------------------->" << std::endl;
	example_gpu_train(ap);

}
