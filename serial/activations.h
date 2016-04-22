#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <vector>

namespace neuralnet
{	
	/*
	 * Standard Activation Functions
	 * Format Description:
	 *		D(): Activation function derivative.
	 *		F(): Activation function.
	 *		operator(): Activation function.
	 */
	template<typename T>
	struct Activation
	{
		virtual inline void D(T &x, T &y) = 0;		
		virtual inline void F(T &x, T &y) = 0;		
		virtual inline void operator()(T &x, T &y) = 0;

		virtual void D(std::vector<T> &x, std::vector<T> &y) = 0;		
		virtual void F(std::vector<T> &x, std::vector<T> &y) = 0;		
		virtual void operator()(std::vector<T> &x, std::vector<T> &y) = 0;
	};

	template<typename T>
	struct Sigmoid: public Activation<T>
	{
		void D(T &x, T &y);
		void F(T &x, T &y);
		void operator()(T &x, T &y);
		
		void D(std::vector<T> &x, std::vector<T> &y);
		void F(std::vector<T> &x, std::vector<T> &y);
		void operator()(std::vector<T> &x, std::vector<T> &y);
	};

	template<typename T>
	struct FSigmoid: public Activation<T>
	{	
		void D(T &x, T &y);
		void F(T &x, T &y);
		void operator()(T &x, T &y);
		
		void D(std::vector<T> &x, std::vector<T> &y);
		void F(std::vector<T> &x, std::vector<T> &y);
		void operator()(std::vector<T> &x, std::vector<T> &y);
	};

	template<typename T>
	struct ReLU: public Activation<T>
	{
		void D(T &x, T &y);
		void F(T &x, T &y);
		void operator()(T &x, T &y);
		
		void D(std::vector<T> &x, std::vector<T> &y);
		void F(std::vector<T> &x, std::vector<T> &y);
		void operator()(std::vector<T> &x, std::vector<T> &y);
	};

}

#endif // ACTIVATIONS_H
