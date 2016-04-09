#ifndef SNEURAL_NET_H
#define SNEURAL_NET_H

#include "../common/Utils.h"
#include "NetConfig.h"

/*
 *
 *
 */

template<
	typename INPUT_T, /* INPUT LAYER DATA TYPE */
	typename HIDDEN_T, /* HIDDEN LAYER DATA TYPE */
	typename OUTPUT_T, /* OUTPUT LAYER DATA TYPE */
	typename ACT_F
>
class SNeuralNet{
public:
	SNeuralNet();
	SNeuralNet(net_arch arch);
	~SNeuralNet();

	/*Initialize methods*/
	void loadExamplesFromFile(std::string filename);// load training examples from file
	void loadExamples(INPUT_T*,OUTPUT_T*);// provide input for training examples
	void addLayer(Layer<HIDDEN_T,ACT_F> layer);

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
	std::vector<int> arch;
	std::vector<Layer<HIDDEN_T,ACT_F>> network;

	/*Training Examples*/
	INPUT_T* training_examples_input;//TODO: flat memory space or 2D arrays?
	OUTPUT_T* training_examples_output;//TODO: flat memory space or 2D arrays?

	/*Training Parameters*/
	int b = 10;
	double threshold = 0.01;
	int max_iterations = 0;
};

template<typename INPUT_T, typename HIDDEN_T, typename OUTPUT_T, typename ACT_F>
SNeuralNet<INPUT_T, HIDDEN_T, OUTPUT_T, ACT_F>::SNeuralNet(){

}

template<typename INPUT_T, typename HIDDEN_T, typename OUTPUT_T, typename ACT_F>
SNeuralNet<INPUT_T, HIDDEN_T, OUTPUT_T,ACT_F>::SNeuralNet(net_arch arch){
	this->arch = arch;
}

template<typename INPUT_T, typename HIDDEN_T, typename OUTPUT_T, typename ACT_F>
SNeuralNet<INPUT_T, HIDDEN_T, OUTPUT_T, ACT_F>::~SNeuralNet(){

}

template<class INPUT_T,class HIDDEN_T, class OUTPUT_T, typename ACT_F>
void SNeuralNet<INPUT_T, HIDDEN_T, OUTPUT_T, ACT_F>::printNetConfig(){
	std::cout<< "b: " << this->b << std::endl;
	std::cout<< "t: " << this->threshold << std::endl;
	std::cout<< "iter: " << this->max_iterations << std::endl;
	for(int i = 0;i<this->arch.size();i++){
		std::cout<< "layer("<<i<<"): " << this->arch[i] << " neurons"<< std::endl;
	}
}


#endif
