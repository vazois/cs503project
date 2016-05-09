#include"common/ArgParser.h"
#include"common/Utils.h"
#include"parallel_gpu/GNNConfig.h"
#include<iostream>

void benchmarkKernels(ArgParser ap){
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);

	s.bench_test_kernels(MMUL,2048,2048,2048,false);
	s.bench_test_kernels(TMMUL,2048,2048,2048,false);
	s.bench_test_kernels(MHPROD,2048,2048,2048, false);
	s.bench_test_kernels(TVECPVEC,2048,2048,2048,false);
}

void nn1(ArgParser ap){
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);

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
	t.lap("Classification Elapsed Time (ms)");
}

void nn2(ArgParser ap){
	gnn_actf::Sigmoid gs;
	gnn::GNeuralNetwork<float,gnn_actf::Sigmoid> s(gs);

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
	int mode = ap.exists(MDARG) ? ap.getUint(MDARG) : 0 ;
	if(mode == 0)	benchmarkKernels(ap);
	else if(mode==1) nn1(ap);
	else if(mode==2) nn2(ap);
}
