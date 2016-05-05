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

void example00(ArgParser ap){
	cudaSetDevice(0);
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);

	if(!ap.exists(FIARG)) vz::error("Please provide input file!!!");
	s.loadExamplesFromFile("data/glass.data");
	std::vector<int> layers;

	layers.push_back(9);
	layers.push_back(27);
	layers.push_back(7);

	s.setBatchSize(1000);
	s.createLayers(layers);
	s.useTranspose(true);
	s.print_weights();
	Time<millis> t;
	t.start();
	s.train();
	t.lap("Training Execution Time(ms)");

}

void example01(ArgParser ap){
	cudaSetDevice(0);
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);

	//if(!ap.exists(FIARG)) vz::error("Please provide input file!!!");
	s.loadExamplesFromFile("data/letters_train.data");
	s.loadTestExamplesFromFile("data/letters_test.data");
	std::vector<int> layers;

	layers.push_back(16);
	layers.push_back(1000);
	layers.push_back(26);

	s.setBatchSize(500);
	s.createLayers(layers);
	s.useTranspose(true);
	s.setLearningRate(0.1);

	Time<millis> t;
	//s.print_weights();
	t.start();
	//for(int i = 0;i<50;i++)
	s.train();
	t.lap("Training Execution Time(ms)");
	s.printConfig();
	//s.print_weights();
	s.classify();
}

void example02(ArgParser ap){
	cudaSetDevice(0);
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);

	s.loadExamplesFromFile("data/letters_train.data");
	s.loadTestExamplesFromFile("data/letters_test.data");
	std::vector<int> layers;

	layers.push_back(16);
	layers.push_back(1000);
	layers.push_back(700);
	layers.push_back(26);

	unsigned int b = ap.getUint(BARG);
	std::cout<<"b:" << b << std::endl;
	s.setBatchSize(b);
	s.createLayers(layers);
	s.useTranspose(true);
	s.setLearningRate(0.2);

	Time<millis> t;
	//s.print_weights();
	t.start();
	for(int i = 0;i<50;i++) s.train();
	t.lap("Training Execution Time(ms)");
	s.printConfig();
	//s.print_weights();
	s.classify();
}

void example03(ArgParser ap){
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<double,gnn_actf::Sigmoid> s(gs);
	s.loadExamplesFromFile("data/letters_train.data");
	std::vector<int> layers;

	/*layers.push_back(16); //INPUT
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
	layers.push_back(26); //OUTPUT

	s.setBatchSize(5000);
	s.createLayers(layers);
	s.useTranspose(true);
	Time<millis> t;
	t.start();
	s.train();
	t.lap("Training Execution Time");*/

	//784 100 10
	//
	//s.bench_test_kernels(TMMUL,1112,912,1231, false);
	//s.bench_test_kernels(TMMUL,3,4,3, true);
	//s.bench_test_kernels(TMMUL,618,722,356, false);
	//s.bench_test_kernels(MHPROD,3,5,4, true);
	//s.bench_test_kernels(TVECPVEC,457,632,710,false);
	//s.bench_test_kernels(MMUL,3,4,2,true);

	//s.bench_test_kernels(MMUL,3,5,7,true);
	//s.bench_test_kernels(TMMUL,3,5,4,true);
	//s.bench_test_kernels(MHPROD,3,5,4, true);

	for(int i =0 ;i<10;i++){
		s.bench_test_kernels(MMUL,1112,912,1231,false);
		s.bench_test_kernels(TMMUL,1112,912,1231,false);
		s.bench_test_kernels(TVECPVEC,1112,912,100,false);
	}
}

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	//example00(ap);
	//example01(ap);
	//example02(ap);
	example03(ap);

}
