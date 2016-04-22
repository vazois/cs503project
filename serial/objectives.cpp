#include <cmath>
#include <vector>
#include <assert.h>
#include "objectives.h"
#include "linalglib.h"

namespace neuralnet
{	
	template<typename T>
	void Objectives<T>::get_yout(linalglib::Vector<T> &x_train, linalglib::Vector<T> &y_out)
	{
		ret = x_train;
		for(int i = 0; i < this->net->numberOfLayer; i++)
		{
			// How to access to i-th layer? 
			// I put it as this->net->Layer[i], temporary.
			ret = this->net->Layer[i]->fcompute(ret);
		}

		y_out = ret;
	}

	template<typename T> 
	T CrossEntropy<T>::evaluateObj(linalglib::Vector<T> &x_train, linalglib::Vector<T> &y_train)
	{
		T ret;
		linalglib::Vector<T> log_y_out(y_train.size());

		y_out = this->get_yout(x_train);
		assert(y_out.size() == y_train.size());
		
		linalglib::log(y_out, log_y_out); // log(vector type data)
		
		// categorical cross entropy
		ret = -1.0*linalglib::dotproduct(log_y_out, y_train);
		
		return ret;
		
	}

	template<typename T> 
	T CrossEntropy<T>::evaluateObjs(linalglib::Matrix<T> &X_train, linalglib::Matrix<T> &Y_train)
	{
		T ret;

		int nRows = X_train.size();
		assert(nRows > 0 && Y_train.size() = nRows);
		int nCols = X_train[0].size();
		assert(nCols > 0);

		for(int i = 0; i < nRows; i++)
		{
			ret = ret + this->evaluateObj(X_train[i], Y_train[i]);
		}

		// Average Categorical Cross Entropy
		ret = ret/nRows;

		return ret;
		
	}




}