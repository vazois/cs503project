#include <fstream>
#include <iomanip>
#include <string>
#include <numeric>
#include <algorithm> 
#include <unistd.h>
#include <pthread.h>
#include <cmath>
#include <chrono>
#include <random>
#include <iostream>
#include <assert.h>
#include <vector>
#include <sstream>

#include "matrixop.h"

using namespace std;

int num_layers;
int *layers_size;
float ***w, **b, **delta, **a, **z, **sigDz;
float *delC_a, ***delC_w;

float **x_train, **y_train;
float **x_test, **y_test;
float **x_val, **y_val;

float lambda;

void allocate_memory()
{

	ifstream f("net.config");
	f >> num_layers;
	layers_size = (int *)malloc( (num_layers) * sizeof(int) );
	for( int i = 0; i < num_layers; i++)
		f >> layers_size[i];
	
	w = (float ***)malloc( (num_layers - 1) * sizeof(float **) ); 
	delC_w = (float ***)malloc( (num_layers - 1) * sizeof(float **) );
	b = (float **)malloc( (num_layers - 1) * sizeof(float *) ); // b idx 0 corresponds to 1st hidden layer
	delta = (float **)malloc( (num_layers - 1) * sizeof(float *) ); // delta idx 0 corresponds to 1st hidden layer
	a = (float **)malloc( (num_layers - 1) * sizeof(float *) ); // a idx 0 corresponds to 1st hidden layer
	z = (float **)malloc( (num_layers - 1) * sizeof(float *) ); // z idx 0 corresponds to 1st hidden layer
	sigDz = (float **)malloc( (num_layers - 1) * sizeof(float *) );
	for( int i = 0 ; i < num_layers - 1; i++)
	{
		w[i] = (float **)malloc( layers_size[i] * sizeof(float *) );
		delC_w[i] = (float **)malloc( layers_size[i] * sizeof(float *) );
		b[i] = (float *)malloc( layers_size[i+1] * sizeof(float));
		delta[i] = (float *)malloc( layers_size[i+1] * sizeof(float));
		a[i] = (float *)malloc( layers_size[i+1] * sizeof(float));
		z[i] = (float *)malloc( layers_size[i+1] * sizeof(float));
		sigDz[i] = (float *)malloc( layers_size[i+1] * sizeof(float));
		for( int j = 0; j < layers_size[i]; j++)
		{
			w[i][j] = (float *)malloc( layers_size[i+1] * sizeof(float) );
			delC_w[i][j] = (float *)malloc( layers_size[i+1] * sizeof(float) );
		}
	}
	delC_a = (float *)malloc( layers_size[num_layers - 1] * sizeof(float) );
	assert(delC_a != NULL);
}


void initializeGlorot()
{
	seed = std::chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);
	uniform_real_distribution<T> distribution(-1.0,1.0);
	float glorotConstant = sqrt(6)/sqrt(nRows + nCols);
	for(int i = 0; i < num_layers - 1; i++)
	{
		for(int j = 0; j < layers_size[i + 1]; j++)
		{
			for(int k = 0; k < layers_size[i]; k++)
				w[i][k][j] = glorotConstant*distribution(generator);
			b[i][j] = 0;
		}
	}
}

void forwardPass(int idx, int dataset_type)// Idx is the sample idx in the data set
{
	switch(dataset_type)
	{
		case 1: mvProdT(w[0], x_train[idx], z[0], layers_size[0], layers_size[1]);
				break;
		case 2: mvProdT(w[0], x_val[idx], z[0], layers_size[0], layers_size[1]);
				break;
		case 3: mvProdT(w[0], x_test[idx], z[0], layers_size[0], layers_size[1]);
				break;
	}
	add(z[0], b[0], z[0], layers_size[1]);
	sigmoid(z[0], a[0], layers_size[1]);
	for( int i = 1; i < num_layers - 2; i++)
	{
		mvProdT(w[i], a[i - 1], z[i], layers_size[i], layers_size[i+1]);
		add(z[i], b[i], z[i], layers_size[i+1]);
		sigmoid(z[i], a[i], layers_size[i+1]);
		dSigmoid(z[i], sigDz[i]);
	}
	// Add the softmax layer
	mvProdT(w[i], a[i - 1], z[i], layers_size[i], layers_size[i+1]);
	add(z[i], b[i], z[i], layers_size[i+1]);
	softmax(z[i], a[i], layers_size[i+1]);
	softmaxD(z[i], sigDz[i]);
}

void backwardPass(int idx)
{
	costFnD(y_train[idx], a[num_layers - 2], delC_a, layers_size[num_layers - 1]);
	hprod(delC_a, sigDz[num_layers - 2], delta[num_layers - 2], layers_size[num_layers - 1]);
	for(int i = num_layers - 3; i > 0; i--)
	{
		float *temp = (float *)malloc(layers_size[i+1] * sizeof(float));
		mvProd(w[i+1], delta[i+1], temp, layers_size[i+1], layers_size[i+2]);
		hprod(temp, sigDz[i], delta[i], layers_size[i+1]);
		for( int j = 0; j < layers_size[i]; j++)
			for( int k = 0; k < layers_size[i+1]; k++)
					delC_w[i][j][k] = ((i > 0) ? a[i-1][k] : x_train[idx][j])*delta[i][j] + lambda;
	}
}

int main(int argc, char *argv[])
{
	allocate_memory();
	
}

