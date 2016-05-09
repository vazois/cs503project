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
		void get_yout(linalglib::Vector<T> &x_train, linalglib::Vector<T> &y_out);
		void get_yout(linalglib::Matrix<T> &X_train, linalglib::Matrix<T> &Y_out);
		virtual T evaluateObj(linalglib::Vector<T> &x_train, linalglib::Vector<T> &y_train) = 0;
		virtual T evaluateObj(linalglib::Matrix<T> &X_train, linalglib::Matrix<T> &Y_train) = 0;		
	};

	template<typename T>
	struct CrossEntropy: public Objectives<T>
	{
		T evaluateObj(linalglib::Vector<T> &x_train, linalglib::Vector<T> &y_train);
		T evaluateObj(linalglib::Matrix<T> &X_train, linalglib::Matrix<T> &Y_train);
	};
}

#endif // OBJECTIVES_H