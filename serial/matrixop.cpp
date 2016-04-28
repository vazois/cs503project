#include <cmath>

void mvProd(float **M, float *x, float *y, int m, int n)
{
	float sum = 0;
	for(int i = 0; i < m; i++)
	{
		sum = 0;		
		for(int j = 0; j < n; j++)
			sum += M[i][j] * x[j];			
		y[i] = sum;
	}
}

void mvProdT(float **M, float *x, float *y, int m, int n)
{
	float sum = 0;
	for(int i = 0; i < n; i++)
	{
		sum = 0;		
		for(int j = 0; j < m; j++)
			sum += M[j][i] * x[j];			
		y[i] = sum;
	}
}

void add(float *x1, float *x2, float* y, int n)
{
	for(int i = 0; i < n; i++)
		y[i] = x1[i] + x2[i];
}

void sigmoid(float *x, float *y, int n)
{
	for(int i = 0; i < n; i++)
		y[i] = 1/(1 + exp(-x[i]));
}

void hprod(float *x1, float *x2, float *y, int n)
{
	for(int i = 0; i < n; i++)
		y[i] = x1[i]*x2[i];
}

void dSigmoid(float *x, float *y, int n)
{
	int temp = 0;
	for(int i = 0; i < n; i++)
	{
		temp = 1/(1 + exp(-x[i]));
		y[i] = temp * (1 - temp);
	}
}

float costFn(float *y_train, float *y_pred, float ***w, float lambda, int m, int n, int l, int n_out)
{
	float cost = 0;
	for(int i = 0; i < n_out; i++)
		cost -= (y_train[i] * log2(y_pred[i]));
	
	float cost_reg = 0;
	for(int i = 0; i < m; i++)
	{
		for(int j = 0; j < n; j++)
		{
			for(int k = 0; k < l; k++)
			{
				cost_reg += (w[i][j][k]*w[i][j][k]);
			}
		}

	}
	cost_reg *= lambda;
	cost += cost_reg;
	return cost;
}

void costFnD(float *y_train, float *y_pred, float *delC_a, int n)
{
	for(int i = 0; i < n; i++)
		delC_a[i] = -y_train[i]/y_pred[i];
}