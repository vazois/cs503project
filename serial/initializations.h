#ifndef INITIALIZATIONS_H
#define INITIALIZATIONS_H

#include "linalglib.h"

/*
 * Standard initialization functions for a layer
 * Format Description:
 *		D(): Activation function derivative.
 *		F(): Activation function.
 *		operator(): Activation function.
 */
namespace neuralnet{

	template<typename T>
	struct Initializer
	{
		unsigned long seed;
		Initializer();

		virtual inline void initialize(linalglib::Matrix<T> &W, linalglib::Vector<T> &b);
	};
	
	template<typename T>
	struct UniformRandomInitializer: public Initializer<T>
	{
		void initialize(linalglib::Matrix<T> &W, linalglib::Vector<T> &b);
	};
	
	template<typename T>
	struct GlorotUniformInitializer: public Initializer<T>
	{
		void initialize(linalglib::Matrix<T> &W, linalglib::Vector<T> &b);
	};

}
#endif //INITIALIZATIONS_H
