#ifndef REGULARIZER_H
#define REGULARIZER_H

#include "linalglib.h"
/*
 * Regularizer of Layer Weight
 */

namespace neuralnet
{
	template<typename T>
	struct Regularizer
	{
		T lambda;
		Layer<T> *ownLayer;

		Regularizer(T lambda);

		virtual void D(linalglib::Vector<T> &x, linalglib::Vector<T> &y, bool isTrain = true) = 0;
		virtual T F(linalglib::Vector<T> &x, bool isTrain) = 0;
		virtual T operator()(linalglib::Vector<T> &x, bool isTrain) = 0;
	};

	template<typename T>
	struct L2: public Regularizer<T>
	{				
		void D(linalglib::Vector<T> &x, linalglib::Vector<T> &y, bool isTrain = true);
		T F(linalglib::Vector<T> &x, bool isTrain);
		T operator()(linalglib::Vector<T> &x, bool isTrain);	
	};
}

#endif // REGULARIZER_H

