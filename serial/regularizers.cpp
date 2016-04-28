#include "regularizers.h"

/*
 * Regularizer of Layer Weight
 */

namespace neuralnet
{
	template<typename T>
	Regularizer<T>::Regularizer(T lambda)
	{
		this->lambda = lambda;
	}

	template<typename T>
	void L2<T>::D(linalglib::Vector<T> &x, linalglib::Vector<T> &y, bool isTrain)
	{
		if(isTrain)
			linalglib::prod(x, 2*this->lambda, y);
		else
			linalglib::zeros(y);
	}

	template<typename T>
	T L2<T>::F(linalglib::Vector<T> &x, bool isTrain)
	// Param isTrain has default value = True
	{	    
	    T ret = 0;		
		if(isTrain)
	    {
	    	linalglib::dprod(x, x, ret);		    
		    ret *= this->lambda;
		}
		return ret;
	}

	template<typename T>
	T L2<T>::operator()(linalglib::Vector<T> &x, bool isTrain)
	// Param isTrain has default value = True
	{
		return L2::F(x, isTrain);
	}
}

