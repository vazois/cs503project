#include "layer.h"
#include "linalglib.h"

namespace neuralnet{

	template <typename WEIGHT_T>
	Layer<WEIGHT_T>::Layer(int n_in, int n_out, Activation<WEIGHT_T> &F, Initializer<WEIGHT_T> &I, Regularizer<WEIGHT_T> &R)
	: W(n_in*n_out), b(n_out), z(n_out, 0.0), y(n_out, 0.0), F(F), I(I), R(R)
	{
			this->n_in = n_in;
			this->n_out = n_out;
			this->F = F;
			this->I = I;
			this->R = R;

			I.initialize(W, b);
	}

	template <typename WEIGHT_T>
	Layer<WEIGHT_T>::~Layer()
	{			
	}

	template <typename WEIGHT_T>
	void Layer<WEIGHT_T>::fcompute(std::vector<WEIGHT_T> &x)
	{
		linalglib::prod(this->W, x, this->z);
		linalglib::add(this->z, this->b, this->z);
		this->F(this->z, this->y);
	}

}
