#include "initializations.h"
#include <chrono>
#include <cmath>
#include <assert.h>
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
    Initializer<T>::Initializer()
	{
		seed = std::chrono::system_clock::now().time_since_epoch().count();	
	}

	template<typename T>
    void UniformRandomInitializer<T>::initialize(std::vector< std::vector<T> > &W, std::vector<T> &b)
	{
		int nRows = W.size();
		assert(nRows > 0);
		int nCols = W[0].size();

		std::default_random_engine generator(this->seed);
		std::uniform_real_distribution<T> distribution(0.0,1.0);
		for(int i = 0; i < nRows; i++)
		{
			assert(W[i].size() == nCols);
			for(int j = 0; j < nCols; j++)
				W[i][j] = distribution(generator);
			b[i] = 0;
		}
	}
	
	template<typename T>
	void GlorotUniformInitializer<T>::initialize(std::vector< std::vector<T> > &W, std::vector<T> &b)
	{
		int nRows = W.size();
		assert(nRows > 0);
		int nCols = W[0].size();

		std::default_random_engine generator(this->seed);
		std::uniform_real_distribution<T> distribution(-1.0,1.0);
		T glorotConstant = sqrt(6)/sqrt(nRows + nCols);
		for(int i = 0; i < nRows; i++)
		{
			assert(W[i].size() == nCols);
			for(int j = 0; j < nCols; j++)
				W[i][j] = glorotConstant*distribution(generator);
			b[i] = 0;
		}
	}

}
