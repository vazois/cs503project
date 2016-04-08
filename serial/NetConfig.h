#include <cmath>
typedef std::vector<int> net_arch;

/*
 * Neural Network Layer
 *
 * */

template<typename OUTPUT_T>
struct Layer{
	unsigned int in_neurons;
	unsigned int out_neurons;

	OUTPUT_T* weights;
	OUTPUT_T* out;

	Layer(unsigned int in_neurons, unsigned int out_neurons){
		this->in_neurons = in_neurons;
		this->out_neurons = out_neurons;

		this->weights = new OUTPUT_T[in_neurons * out_neurons];
		this->out = new OUTPUT_T[out_neurons];
	}

	~Layer(){
		delete this->weights;
		delete this->out;
	}

	template<typename FuncType>
	OUTPUT_T activate_layer(OUTPUT_T input, FuncType Func){
		return Func(input);
	}

};

/*
 * Standard Activation Functions
 *
 * */

namespace stdaf{
	template<typename OUTPUT_T>
	OUTPUT_T sigmoid(OUTPUT_T x){
		return 1/(1 + exp(-x));
	}

	template<typename OUTPUT_T>
	OUTPUT_T fsigmoid(OUTPUT_T x){
		return x/(1.0 + fabs(x));
	}

	template<typename OUTPUT_T>
	OUTPUT_T test(OUTPUT_T x){
		return x + x;
	}
}
