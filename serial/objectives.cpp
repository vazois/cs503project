#include <assert.h>
#include "objectives.h"
#include "linalglib.h"

namespace neuralnet
{
	template<typename T> 
	T CrossEntropy<T>::evaluateObj(linalglib::Vector<T> &x_train, linalglib::Vector<T> &y_train, bool isTrain)
	// Param isTrain has default value = True
	{
		T ret;
		linalglib::Vector<T> y_out(y_train.size());
		linalglib::Vector<T> log_y_out(y_train.size());

		this->net->evaluate(x_train, y_out);
		assert(y_out.size() == y_train.size());
		
		linalglib::log(y_out, log_y_out); // log(vector type data)
		
		// categorical cross entropy
		ret = -1*linalglib::dprod(log_y_out, y_train);
		
		return ret;		
	}

	template<typename T> 
	T CrossEntropy<T>::evaluateObj(linalglib::Matrix<T> &X_train, linalglib::Matrix<T> &Y_train, bool isTrain)
	// Param isTrain has default value = True
	{
		T ret;
		int nRows = X_train.size();
		assert(nRows > 0 && Y_train.size() = nRows);
		int nCols = X_train[0].size();
		assert(nCols > 0);

		for(int i = 0; i < nRows; i++)
		{
			ret += this->evaluateObj(X_train[i], Y_train[i]);
		}

		return ret;		
	}

}