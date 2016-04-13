#include <cmath>
typedef std::vector<int> net_arch;

typedef int (*Fun)(int,int);

/*
 * Neural Network Layer Abstraction
 */

template<typename LAYER_T, typename ACT_F>
struct Layer{
	unsigned int input_nn;/*Number of input neurons*/
	unsigned int output_nn;/*Number of output neurons*/

	LAYER_T* W_j;/*Input weight matrix*/
	LAYER_T* alpha_jj;/*Ouput neuron vector*/
	LAYER_T* delta_jj;/*Propagated error vector*/
	LAYER_T* D_j; /*Derivative of weights*/

	ACT_F F;/*Activation Function*/

	/*
	 * Layer Constructor
	 * (1): Number of input neurons.
	 * (2): Number of output neurons.
	 * (3): Custom activation function. Can be applied on the value of the linear combination at each neuron.
	 */
	Layer(unsigned int input_nn, unsigned int output_nn, ACT_F F){
		this->input_nn = input_nn;
		this->output_nn = output_nn;

		this->F = F;

		this->W_j = new LAYER_T[this->input_nn * this->output_nn];
		this->D_j = new LAYER_T[this->input_nn * this->output_nn];
		this->alpha_jj = new LAYER_T[this->output_nn];
		this->delta_jj = new LAYER_T[this->output_nn];
	}

	/*
	 * Struct destructor. Free memory.
	 */
	~Layer(){
		delete this->W_j;
		delete this->alpha_jj;
	}

	/*
	 * Test activation function.
	 */
	inline LAYER_T operator()(LAYER_T x){
		return F(x);
	}

	/*
	 * Feed Forward Computation
	 * (1): Previous layer activation vector.
	 */
	void ffcompute(LAYER_T* alpha_jin){
		// alpha_jj = W_j * alpha_jin
		// alpha_jj = F(alpha_jj)
	}

	/*
	 * Back Propagation Computation
	 * (1): Next layer delta values.
	 * (2): Output weights
	 */
	void bpcompute(LAYER_T* delta_jout, LAYER_T* W_jout_T, LAYER_T* alpha_jin){
		//delta_jj = (W_jout)_T . delta_jout .* alpha_jj .* (1 - alpha_jj)
		//D_j = delta_jj . (alpha_jin)
		//W_j = W_j + n.D_j
	}
};

/*
 * Standard Activation Functions
 * Format Description:
 *		D(): Activation function derivative.
 *		F(): Activation function.
 *		operator(): Activation function.
 */
namespace stdaf{
	struct Sigmoid{
		template<typename T>
		inline T D(T x){
			return x*(1-x);
		}

		template<typename T>
		inline T F(T x){
			return 1/(1 + exp(-x));
		}

		template<typename T>
		inline T operator()(T x){
			return 1/(1 + exp(-x));
		}
	};

	struct FSigmoid{
		template<typename T>
		inline T D(T x){
				return x*(1-x);
		}

		template<typename T>
		inline T F(T x){
				return 1/(1 + exp(-x));
		}

		template<typename T>
		inline T operator()(T x){
			return x/(1.0 + fabs(x));
		}
	};
}


