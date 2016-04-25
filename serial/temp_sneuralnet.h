#ifndef SNEURAL_NET_H
#define SNEURAL_NET_H

#include "../common/Utils.h"
#include "NetConfig.h"

/*
 *
 *
 */

template<
	typename DATA_T, /* INPUT LAYER DATA TYPE */
	typename ACT_F /* ACTIVATION FUNCTION */
>
class SNeuralNet{
public:
	SNeuralNet();
	SNeuralNet(net_arch arch,ACT_F F);
	~SNeuralNet();

	/*Initialize methods*/
	void init();
	void loadExamplesFromFile(std::string filename);// load training examples from file
	void loadExamples(DATA_T*,DATA_T*);// provide input for training examples
	void addLayer(Layer<DATA_T,ACT_F> layer);

	/*Training Configuration*/
	void setBatchSize(int b){ this->b = b; };//default value 10
	void setTrainErrorThreshold(double threshold){ this->threshold = threshold; };// default value 0.0001
	void setMaxIterations(int max_iterations){ this->max_iterations = max_iterations; };

	/*training methods*/
	void train();//

	/*Helper Methods*/
	void printNetConfig();

private:
	/*Network Architecture*/
	net_arch arch;
	std::vector<Layer<DATA_T,ACT_F>*> network;
	ACT_F F;

	/*Training Examples*/
	DATA_T* training_examples_input;//TODO: flat memory space or 2D arrays?
	DATA_T* training_examples_output;//TODO: flat memory space or 2D arrays?

	/*Training Parameters*/
	int b = 10;
	double threshold = 0.01;
	int max_iterations = 0;
};

template<typename DATA_T, typename ACT_F>
SNeuralNet<DATA_T,ACT_F>::SNeuralNet(){

}

template<typename DATA_T, typename ACT_F>
SNeuralNet<DATA_T,ACT_F>::SNeuralNet(net_arch arch, ACT_F F){
	this->arch = arch;
	this->F = F;
}

template<typename DATA_T, typename ACT_F>
SNeuralNet<DATA_T,ACT_F>::~SNeuralNet(){

}

template<typename DATA_T, typename ACT_F>
void SNeuralNet<DATA_T,ACT_F>::init(){
	for(int i = 1;i<this->arch.size();i++){
		network.push_back(new Layer<DATA_T,ACT_F>(this->arch[i-1],this->arch[i],this->F));
	}
}

template<typename DATA_T, typename ACT_F>
void SNeuralNet<DATA_T,ACT_F>::printNetConfig(){
	std::cout<< "b: " << this->b << std::endl;
	std::cout<< "t: " << this->threshold << std::endl;
	std::cout<< "iter: " << this->max_iterations << std::endl;
	for(int i = 0;i<this->arch.size();i++){
		std::cout<< "layer("<<i<<"): " << this->arch[i] << " neurons"<< std::endl;
	}
}


#endif
