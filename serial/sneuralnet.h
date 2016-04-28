#ifndef SNEURALNET_H
#define SNEURALNET_H

#include "linalglib.h"

namespace neuralnet
{
	template<typename WEIGHT_T>
	struct SNeuralNet
	{
		linalglib::Vector<Layer<WEIGHT_T> > layers;
		Objectives<WEIGHT_T> &objective;
		Optimizer<WEIGHT_T> &optimizer;
		int n_in;
		
		SNeuralNet(int n_in, Objectives<WEIGHT_T> &objective, Optimizer<WEIGHT_T> &optimizer);
		
		void addLayer(int n_out, Activation<WEIGHT_T>& F, Initializer<WEIGHT_T>& I, Regularizer<WEIGHT_T>& R);
		void addLayer(Layer<WEIGHT_T>& layer);
		void train();
		void evaluate(linalglib::Vector<T> &x_train, linalglib::Vector<T> &y_out);
		void evaluate(linalglib::Matrix<T> &X_train, linalglib::Matrix<T> &Y_out);
	};
}
#endif // SNEURALNET_H