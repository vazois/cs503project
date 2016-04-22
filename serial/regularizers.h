#ifndef REGULARIZER_H
#define REGULARIZER_H

#include <cmath>
#include <vector>
/*
 * Regularizer of Layer Weight
 */

namespace neuralnet
{
	template<typename T>
	struct Regularizer
	{

		T lambda;  // initialize?

		Regularizer(T lambda)
		{
			this->lambda = lambda;
		}
			



		virtual void D(std::vector<T> &x, std::vector<T> &y, bool isTrain) = 0;


		virtual T F(std::vector<T> &x, bool isTrain) = 0;


		virtual T operator()(std::vector<T> &x, bool isTrain) = 0;
	};

	template<typename T>
	struct L2: public Regularizer<T>
	{
				
		void D(std::vector<T> &x, std::vector<T> &y, bool isTrain);


		T F(std::vector<T> &x, bool isTrain);


		T operator()(std::vector<T> &x, bool isTrain);	
	};

}

#endif // REGULARIZER_H

