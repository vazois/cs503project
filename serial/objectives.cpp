#include <cmath>
#include <vector>
#include <assert.h>
#include "objectives.h"

namespace neuralnet
{	
	template<typename T>
	void Objectives<T>::get_yout(std::vector<T> &x_train, std::vector<T> &y_out)
	{
		ret = this->net->Layer[0]->fcompute(x_train);
		for(int i=1; i < this->net->numberOfLayer; i++)
		{
			ret = this->net->Layer[i]->fcompute(ret);
		}

		y_out = ret;
	}

	template<typename T> 
	T CrossEntropy<T>::evaluateObj(std::vector<T> &x_train, std::vector<T> &y_train)
	{
		T ret = 0;

		y_out = this->get_yout(x_train);
		assert(y_out.size() == y_train.size());
		
		for(int i=0; i < y_out.size(); i++)
		{
			ret = ret - log(y_out[i])*y_train[i];
		}

		return ret;
		

	}




}