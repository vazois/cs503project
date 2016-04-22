#include "regularizers.h"

/*
 * Regularizer of Layer Weight
 */

namespace neuralnet
{
	template<typename T>
	void L2<T>::D(std::vector<T> &x, std::vector<T> &y, bool isTrain)
	{
		if (isTrain)
		{
			int i;

			for (i = 0;i < y.size();i++)
			{
		    	y[i] = 2*this->lambda*x[i];
		    }
		}
	}

	template<typename T>
	T L2<T>::F(std::vector<T> &x, bool isTrain)
	{
	    
	    T ret = 0;
		
		if (isTrain)
	    {
	    	int i;    
		    for (i=0;i<x.size();i++)
		    {
		        ret = ret + pow(x[i],2);
		    }
		}

	    return this->lambda*ret;
	}

	template<typename T>
	T L2<T>::operator()(std::vector<T> &x, bool isTrain)
	{
		return L2::F(x, isTrain);
	}
}

