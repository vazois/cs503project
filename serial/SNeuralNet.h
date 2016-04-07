#ifndef SNEURAL_NET
#define SNEURAL_NET

#include "../common/Utils.h"

typedef std::vector<int> net_arch;

/*
 * -
 *
 * */

template<class INPUT_T,class HIDDEN_T, class OUTPUT_T>
class SNeuralNet{
public:
	SNeuralNet();
	SNeuralNet(net_arch arch);
	~SNeuralNet();

	/*Initialize methods*/
	void load_train_examples(std::string filename);// load training examples from file
	void init_train_examples(INPUT_T*,OUTPUT_T*);// provide input for training examples

	void set_batch_size(int b){ this->b = b; };//default value 10
	void set_error_threshold(double threshold){ this->threshold = threshold; };// default value 0.0001
	void set_max_iterations(int max_iterations){ this->max_iterations = max_iterations; };

	/*training methods*/
	void train();// Iterations should be > 0, if == 0 then

	/*Helper Methods*/
	void print_configuration();

private:
	std::vector<int> arch;
	INPUT_T* training_examples_input;//TODO: flat memory space or 2D arrays?
	OUTPUT_T* training_examples_output;//TODO: flat memory space or 2D arrays?
	HIDDEN_T* nn_weights;

	int b = 10;
	double threshold = 0.0001;
	int max_iterations = 0;

	int input_layer = 0;
	int output_layer = 0;
};

template<class INPUT_T,class HIDDEN_T, class OUTPUT_T>
SNeuralNet<INPUT_T, HIDDEN_T, OUTPUT_T>::SNeuralNet(){

}

template<class INPUT_T,class HIDDEN_T, class OUTPUT_T>
SNeuralNet<INPUT_T, HIDDEN_T, OUTPUT_T>::SNeuralNet(net_arch arch){
	this->arch = arch;
}

template<class INPUT_T,class HIDDEN_T, class OUTPUT_T>
SNeuralNet<INPUT_T, HIDDEN_T, OUTPUT_T>::~SNeuralNet(){

}

template<class INPUT_T,class HIDDEN_T, class OUTPUT_T>
void SNeuralNet<INPUT_T, HIDDEN_T, OUTPUT_T>::print_configuration(){
	std::cout<< "b: " << this->b << std::endl;
	std::cout<< "t: " << this->threshold << std::endl;
	std::cout<< "iter: " << this->max_iterations << std::endl;
	for(int i = 0;i<this->arch.size();i++){
		std::cout<< "layer("<<i<<"): " << this->arch[i] << " neurons"<< std::endl;
	}
}


#endif
