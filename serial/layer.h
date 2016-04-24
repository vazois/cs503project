#ifndef LAYER_H
#define LAYER_H

#include "activations.h"
#include "initializations.h"
#include "regularizers.h"
#include "linalglib.h"

namespace neuralnet
{
	/*
	 * Neural Network Layer Abstraction
	 */
	template<typename WEIGHT_T>
	struct Layer
	{
		int n_in;					/*Number of input neurons*/
		int n_out;					/*Number of output neurons*/

		linalglib::Matrix<WEIGHT_T> W;					/*Input weight linalglib::Matrix*/
		linalglib::Vector<WEIGHT_T> b;					/*Input bias linalglib::Vector*/
		linalglib::Vector<WEIGHT_T> z;					/*Pre-activation linalglib::Vector*/
		linalglib::Vector<WEIGHT_T> y;					/*Output linalglib::Vector*/
		// std::linalglib::Vector<WEIGHT_T> del;		/*Propagated error linalglib::Vector*/
		// std::linalglib::Vector<WEIGHT_T> D_W;		/*Derivative w.r.t. weights*/
		// std::linalglib::Vector<WEIGHT_T> D_b;		/*Derivative w.r.t. biases*/

		Activation<WEIGHT_T> &F;							/*Activation Object*/
		Initializer<WEIGHT_T> &I;						/*Initialization Object*/
		Regularizer<WEIGHT_T> &R;							/*Regularizer Object*/

		/*
		 * Layer Constructor
		 * (1): Number of input neurons.
		 * (2): Number of output neurons.
		 * (3): Custom activation object for layer.
		 * (4): Custom initialization function for layer.
		 * (5): Custom regularizer for layer.
		 */
		Layer(int n_in, int n_out, Activation<WEIGHT_T> &F, Initializer<WEIGHT_T> &I, Regularizer<WEIGHT_T> &R);

		~Layer();

		/*
		 * Forward Computation
		 * (1): Input linalglib::Vector.
		 */
		void fcompute(linalglib::Vector<WEIGHT_T> &x);		

		/*
		 * Back Propagation Computation
		 * (1): Next layer delta values.
		 * (2): Output weights
		 */
		// void bpcompute(WEIGHT_T* delta_jout, WEIGHT_T* W_jout_T, WEIGHT_T* alpha_jin){
			//delta_jj = (W_jout)_T . delta_jout .* alpha_jj .* (1 - alpha_jj)
			//D_j = delta_jj . (alpha_jin)
			//W_j = W_j + n.D_j
		// }
	};

}

#endif // LAYER_H