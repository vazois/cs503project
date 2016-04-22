#ifndef OBJECTIVES_H
#define OBJECTIVES_H

#include <vector>
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
		void get_yout(linalglib::Vector<T> &x_train, linalglib::Vector<T> &y_out) = 0;
		virtual T evaluateObj(linalglib::Vector<T> &x_train, linalglib::Vector<T> &y_train) = 0;
		virtual T evaluateObjs(linalglib::Matrix<T> &X_train, linalglib::Matrix &Y_train) = 0;
		
	};

	template<typename T>
	struct CrossEntropy: public Objectives<T>
	{
		//void get_yout(std::vector<T> &x_train, std::vector<T> &y_out);
		T evaluateObj(linalglib::Vector<T> &x_train, linalglib::Vector<T> &y_train);
		T evaluateObjs(linalglib::Matrix<T> &X_train, linalglib::Matrix &Y_train);
	};

	template<typename T>
	struct LeastMeanSquare: public Objectives<T>
	{	
		//void get_yout(std::vector<T> &x_train, std::vector<T> &y_out);
		T evaluateObj(linalglib::Vector<T> &x_train, linalglib::Vector<T> &y_train);
		T evaluateObjs(linalglib::Matrix<T> &X_train, linalglib::Matrix &Y_train);
	};

}

#endif // OBJECTIVES_H