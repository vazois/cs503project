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
	cudaSetDevice(CUDA_DEVICE);
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);

	s.loadExamplesFromFile("data/glass.data");
	std::vector<int> layers;

	layers.push_back(9);
	layers.push_back(27);
	layers.push_back(7);

	s.setBatchSize(10);
	s.createLayers(layers);
	s.useTranspose(true);
	//s.print_weights();
	Time<millis> t;
	t.start();
	s.train();
	t.lap("Training Execution Time(ms)");

}

void example01(ArgParser ap){
	cudaSetDevice(CUDA_DEVICE);
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);

	//if(!ap.exists(FIARG)) vz::error("Please provide input file!!!");
	s.loadExamplesFromFile("data/glass.data");
	std::vector<int> layers;

	layers.push_back(16);
	layers.push_back(100);
	layers.push_back(26);

	s.setBatchSize(100);
	s.createLayers(layers);
	s.useTranspose(true);
	s.setLearningRate(0.1);

	Time<millis> t;
	//s.print_weights();
	t.start();
	for(int i = 0;i<50;i++) s.train();
	t.lap("Training Execution Time(ms)");
}

void example02(ArgParser ap){
	cudaSetDevice(CUDA_DEVICE);
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);

	s.loadExamplesFromFile("data/letters_train.data");
	s.loadTestExamplesFromFile("data/letters_test.data");
	std::vector<int> layers;

	layers.push_back(16);
	layers.push_back(500);
	//layers.push_back(700);
	//layers.push_back(700);
	layers.push_back(26);

	unsigned int b = ap.getUint(BARG);
	std::cout<<"b:" << b << std::endl;
	s.setBatchSize(b);
	s.createLayers(layers);
	s.useTranspose(true);
	s.setLearningRate(0.3);

	Time<millis> t;
	//s.print_weights();
	unsigned int iterations = 100;
	t.start();
	for(int i = 0;i<iterations;i++) s.train();
	s.printConfig(t.lap("Training Execution Time(ms)")/iterations);
	//s.print_weights();
	s.classify();
}

void example03(ArgParser ap){
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);
	gnn_actf::FSigmoid fgs;
	//gnn::GNeuralNetwork<double,gnn_actf::FSigmoid> s(fgs);

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

	Time<millis> t;
	t.start();
	for(int i =0 ;i<1;i++){
		s.bench_test_kernels(MMUL,2048,2048,2048,false);
		s.bench_test_kernels(TMMUL,2048,2048,2048,false);
		s.bench_test_kernels(MHPROD,2048,2048,2048, false);
		s.bench_test_kernels(TVECPVEC,2048,2048,2048,false);
	}
	t.lap();
}

void example04(ArgParser ap){
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);

	//gnn_actf::FSigmoid fgs;
	//gnn::GNeuralNetwork<float,gnn_actf::FSigmoid> s(fgs);

	Time<millis> t;
	t.start();
	s.loadExamplesFromFile("../mnist_train.csv");
	s.loadTestExamplesFromFile("../mnist_test.csv");
	t.lap("Read Train and Test Data");

	std::vector<int> layers;
	layers.push_back(784);//Input Layer
	layers.push_back(1024);//Hidden Layer
	layers.push_back(10);//Output Layer

	unsigned int iterations = ap.exists(IARG) ? ap.getUint(IARG) : 50 ;
	unsigned int b = ap.exists(BARG) ? ap.getUint(BARG) : 100 ;
	float r = ap.exists(DARG) ? ap.getFloat(DARG) : 0.1 ;
	std::cout<<"i:" << iterations << std::endl;
	std::cout<<"b:" << b << std::endl;
	std::cout<<"r:" << r << std::endl;

	s.setBatchSize(b);
	s.useTranspose(true);
	s.setLearningRate(r);
	s.createLayers(layers);
	if(!s.validateInput()) vz::error("Input + Ouput Neurons != number of features");

	std::cout<<"Training...";
	t.start();
	for(int i = 0;i<iterations;i++){ s.train(); } std::cout << std::endl;
	s.printConfig(t.lap("Training Execution Time(ms)")/iterations);

	t.start();
	std::cout<<"Computing Classification Accuracy..." << std::endl;
	s.classify();
	t.lap("Classification Elapsed Time");
}

void example05(ArgParser ap){
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);

	//gnn_actf::FSigmoid fgs;
	//gnn::GNeuralNetwork<float,gnn_actf::FSigmoid> s(fgs);

	Time<millis> t;
	t.start();
	s.loadExamplesFromFile("../mnist_train.csv");
	s.loadTestExamplesFromFile("../mnist_test.csv");
	t.lap("Read Train and Test Data");

	std::vector<int> layers;
	layers.push_back(784);//Input Layer
	layers.push_back(1024);//Hidden Layer 1
	layers.push_back(1024);//Hidden Layer 2
	layers.push_back(10);//Output Layer

	unsigned int iterations = ap.exists(IARG) ? ap.getUint(IARG) : 50 ;
	unsigned int b = ap.exists(BARG) ? ap.getUint(BARG) : 100 ;
	float r = ap.exists(DARG) ? ap.getFloat(DARG) : 0.1 ;
	std::cout<<"i:" << iterations << std::endl;
	std::cout<<"b:" << b << std::endl;
	std::cout<<"r:" << r << std::endl;

	s.setBatchSize(b);
	s.useTranspose(true);
	s.setLearningRate(r);
	s.createLayers(layers);
	if(!s.validateInput()) vz::error("Input + Ouput Neurons != number of features");

	std::cout<<"Training...";
	t.start();
	for(int i = 0;i<iterations;i++){ s.train(); } std::cout << std::endl;
	s.printConfig(t.lap("Training Execution Time(ms)")/iterations);

	t.start();
	std::cout<<"Computing Classification Accuracy..." << std::endl;
	s.classify();
	t.lap("Classification Elapsed Time");
}

int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);
	//example00(ap);
	//example01(ap);
	//example02(ap);
	//example03(ap);
	example04(ap);
	//example05(ap);
}
