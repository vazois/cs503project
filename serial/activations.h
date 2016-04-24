#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "linalglib.h"

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

		virtual void D(linalglib::Vector<T> &x, linalglib::Vector<T> &y) = 0;		
		virtual void F(linalglib::Vector<T> &x, linalglib::Vector<T> &y) = 0;		
		virtual void operator()(linalglib::Vector<T> &x, linalglib::Vector<T> &y) = 0;
	};

	template<typename T>
	struct Sigmoid: public Activation<T>
	{
		void D(T &x, T &y);
		void F(T &x, T &y);
		void operator()(T &x, T &y);
		
		void D(linalglib::Vector<T> &x, linalglib::Vector<T> &y);
		void F(linalglib::Vector<T> &x, linalglib::Vector<T> &y);
		void operator()(linalglib::Vector<T> &x, linalglib::Vector<T> &y);
	};

	template<typename T>
	struct FSigmoid: public Activation<T>
	{	
		void D(T &x, T &y);
		void F(T &x, T &y);
		void operator()(T &x, T &y);
		
		void D(linalglib::Vector<T> &x, linalglib::Vector<T> &y);
		void F(linalglib::Vector<T> &x, linalglib::Vector<T> &y);
		void operator()(linalglib::Vector<T> &x, linalglib::Vector<T> &y);
	};

	template<typename T>
	struct ReLU: public Activation<T>
	{
		void D(T &x, T &y);
		void F(T &x, T &y);
		void operator()(T &x, T &y);
		
		void D(linalglib::Vector<T> &x, linalglib::Vector<T> &y);
		void F(linalglib::Vector<T> &x, linalglib::Vector<T> &y);
		void operator()(linalglib::Vector<T> &x, linalglib::Vector<T> &y);
	};

}

#endif // ACTIVATIONS_H
