#include "sneuralnet.h"

namespace neuralnet
{
	template<typename WEIGHT_T>
	SNeuralNet<T>::SNeuralNet(int n_in, Objectives<WEIGHT_T> &objective, Optimizer<WEIGHT_T> &optimizer)
	:n_in(n_in), objective(objective), optimizer(optimizer)
	{
	}
	
	template<typename WEIGHT_T>
	SNeuralNet<T>::addLayer(int n_out, Activation<WEIGHT_T>& F, Initializer<WEIGHT_T>& I, Regularizer<WEIGHT_T>& R)
	{
		if( this->layers.size() == 0 )
			this->layers.push_back(Layer<WEIGHT_T>(this-n_in, n_out, F, I, R));
		else
		{
			Layer<WEIGHT_T> &prevLayer = this->layers[this->layers.size() - 1];
			this->layers.push_back(Layer<WEIGHT_T>(prevLayer->n_out, n_out, F, I, R));
		}
	}
	
	template<typename WEIGHT_T>
	SNeuralNet<T>::addLayer( Layer<WEIGHT_T>& layer )
	{
		Layer<WEIGHT_T> &prevLayer = this->layers[this->layers.size() - 1];
		assert( prevLayer->n_out == layer->n_in );
		this->layers.push_back(layer);
	}
	
	template<typename WEIGHT_T>
	SNeuralNet<T>::train( )
	{
		// Fill it when you can
	}

	template<typename T>
	void Objectives<T>::evaluate(linalglib::Vector<T> &x_train, linalglib::Vector<T> &y_out)
	{
		linalglib::Vector<T> *ret = &x_train;
		for(int i = 0; i < this->net->layers.size(); i++)			
		{
			this->net->layers[i]->fcompute(*ret);
			ret = &(this->net->layers[i]->y);
		}

		y_out = *ret;
	}

	template<typename T>
	void Objectives<T>::evaluate(linalglib::Matrix<T> &X_train, linalglib::Matrix<T> &Y_out)
	{
		int batch_size = X_train.size();
		assert(Y_out.size() == batch_size);
		for(int i = 0; i < batch_size; i++)
			evaluate(X_train[i], Y_out[i]);
	}
	
}