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
		unsigned seed;
		Initializer()
		{
			seed = chrono::system_clock::now().time_since_epoch().count();	
		}

		virtual inline void initialize(std::vector< std::vector<T> > &W, std::vector<T> &b);
	};
	
	struct UniformRandomInitializer: public Initializer{};
	
	struct GlorotRandomInitializer: public Initializer{};

}
#endif //INITIALIZATIONS_H

