#ifndef REGULARIZER_H
#define REGULARIZER_H

/*
 * Regularizer of Layer Weight
 */

namespace neuralnet
{
	template<typename T>
	struct Regularizer
	{
		template<typename T>
		T lambda;  // initialize?

		Regularizer(T lambda){
			this->lambda = lambda;
		}

		template<typename T>
		virtual std::vector<T> D(std::vector<T> &x, std::vector<T> &y, bool isTrain) = 0;

		template<typename T>
		virtual T F(std::vector<T> &x, bool isTrain) = 0;

		template<typename T>
		virtual T operator()(std::vector<T> &x, bool isTrain) = 0;
	};

	struct L2: public Regularizer
	{
		template<typename T>
		T lambda;
		
		Regularizer(T lambda);

		template<typename T>
		std::vector<T> D(std::vector<T> &x, std::vector<T> &y, bool isTrain);

		template<typename T>
		T F(std::vector<T> &x, bool isTrain);

		template<typename T>
		T operator()(std::vector<T> &x, bool isTrain);	
	};

}

#endif // REGULARIZER_H

