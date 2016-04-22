#include "initializations.h"

/*
 * Standard initialization functions for a layer
 * Format Description:
 *		D(): Activation function derivative.
 *		F(): Activation function.
 *		operator(): Activation function.
 */
namespace neuralnet{

	template<typename T>
    void UniformRandomInitializer::initialize(std::vector< std::vector<T> > &W, std::vector<T> &b)
	{
		int nRows = W.size();
		assert(nRows > 0);
		int nCols = W[0].size();

		default_random_engine generator (seed);
		uniform_real_distribution<T> distribution (0.0,1.0);
		for(int i = 0; i < nRows; i++)
		{
			assert(W[i].size() == nCols);
			for(int j = 0; j < nCols; j++)
				W[i][j] = distribution(generator);
			b[i] = 0;
		}
	}
	
	template<typename T>
	void GlorotRandomInitializer::initialize(std::vector< std::vector<T> > &W, std::vector<T> &b)
	{
		int nRows = W.size();
		assert(nRows > 0);
		int nCols = W[0].size();

		default_random_engine generator (seed);
		uniform_real_distribution<T> distribution (-1.0,1.0);
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

