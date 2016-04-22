#include "regularizers.h"
#include <cmath>

/*
 * Regularizer of Layer Weight
 */

namespace neuralnet
{
	template<typename T>
	void L2<T>::D(std::vector<T> &x, std::vector<T> &y, bool isTrain)
	{
		for (int i = 0;i < y.size();i++)
			y[i] = isTrain ? (2*this->lambda*x[i]) : 0;
	}

	template<typename T>
	T L2<T>::F(std::vector<T> &x, bool isTrain)
	{	    
	    T ret = 0;		
		if(isTrain)
	    {    
		    for (int i = 0; i < x.size(); i++)
		        ret += pow(x[i],2);
		    ret *= this->lambda;
		}
		return ret;
	}

	template<typename T>
	T L2<T>::operator()(std::vector<T> &x, bool isTrain)
	{
		return L2::F(x, isTrain);
	}
}

