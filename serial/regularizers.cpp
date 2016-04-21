#include <cmath>
#include "regularizer.h"

/*
 * Regularizer of Layer Weight
 */

namespace neuralnet
{
	template<typename T>
	void L2::D(std::vector<T> &x, std::vector<T> &y, bool isTrain)
	{
		if (isTrain)
		{
			int i;
			vector<T> y;
			y.reserve(x.size());

			for (i=0;i<y.size();i++)
			{
		    	y[i] = 2*lambda*x[i];
		    }
		}
	}

	template<typename T>
	T L2::F(std::vector<T> &x, bool isTrain)
	{
	    
	    T ret=0;
		
		if (isTrain)
	    {
	    	int i;    
		    for (i=0;i<x.size();i++)
		    {
		        ret = ret + pow(x[i],2);
		    }
		}

	    return lambda*ret;
	}

	template<typename T>
	T L2::operator()(std::vector<T> &x, bool isTrain)
	{
		return L2::F(x, isTrain);
	}
}

