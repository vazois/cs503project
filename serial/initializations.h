#ifndef INITIALIZATIONS_H
#define INITIALIZATIONS_H

#include <cmath>
#include <chrono>
#include <assert.h>
#include <vector>
#include <random>
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

		virtual inline void initialize(std::vector< std::vector<T> > &W, std::vector<T> &b);
	};
	
	template<typename T>
	struct UniformRandomInitializer: public Initializer<T>
	{
		void initialize(std::vector< std::vector<T> > &W, std::vector<T> &b);
	};
	
	template<typename T>
	struct GlorotUniformInitializer: public Initializer<T>
	{
		void initialize(std::vector< std::vector<T> > &W, std::vector<T> &b);
	};

}
#endif //INITIALIZATIONS_H

