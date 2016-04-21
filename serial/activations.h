#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <cmath>

namespace neuralnet{
	
	/*
	 * Standard Activation Functions
	 * Format Description:
	 *		D(): Activation function derivative.
	 *		F(): Activation function.
	 *		operator(): Activation function.
	 */

	struct Activation
	{
		template<typename T>
		virtual inline T D(T x) = 0;

		template<typename T>
		virtual inline T F(T x) = 0;

		template<typename T>
		virtual inline T operator()(T x) = 0;
	};

	struct Sigmoid: public Activation
	{
		template<typename T>
		inline T D(T x)
		{
			return F(x)*(1-F(x));
		}

		template<typename T>
		inline T F(T x)
		{
			return 1/(1 + exp(-x));
		}

		template<typename T>
		inline T operator()(T x)
		{
			return F(x);
		}
	};

	struct FSigmoid: public Activation
	{
		template<typename T>
		inline T D(T x)
		{
				return 1.0/pow(1.0 + std::abs(x),2);
		}

		template<typename T>
		inline T F(T x)
		{
				return x/(1.0 + std::abs(x));
		}

		template<typename T>
		inline T operator()(T x)
		{
			return F(x);
		}
	};

	struct ReLu: public Activation
	{
		template<typename T>
		inline T D(T x)
		{
			return (x >= 0.0 ? 1.0 : 0.0);
		}

		template<typename T>
		inline T F(T x)
		{
			return (x >= 0.0 ? x : 0.0);
		}

		template<typename T>
		inline T operator()(T x)
		{
			return F(x);
		}
	};

}

#endif // ACTIVATIONS_H

