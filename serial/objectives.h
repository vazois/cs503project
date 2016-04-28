#ifndef OBJECTIVES_H
#define OBJECTIVES_H

#include "linalglib.h"
#include "sneuralnet.h"

namespace neuralnet
{	
	/*
	 * Objectives 
	 * Format Description:
	 *		evaluateObj(): Objective function.
	 */
	template<typename T>
	struct Objectives
	{		
		SNeuralNet *net;
		virtual T evaluateObj(linalglib::Vector<T> &x_train, linalglib::Vector<T> &y_train, bool isTrain = true) = 0;
		virtual T evaluateObj(linalglib::Matrix<T> &X_train, linalglib::Matrix<T> &Y_train, bool isTrain = true) = 0;		
	};

	template<typename T>
	struct CrossEntropy: public Objectives<T>
	{
		T evaluateObj(linalglib::Vector<T> &x_train, linalglib::Vector<T> &y_train, bool isTrain = true);
		T evaluateObj(linalglib::Matrix<T> &X_train, linalglib::Matrix<T> &Y_train, bool isTrain = true);
	};
}

#endif // OBJECTIVES_H