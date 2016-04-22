#include <cmath>
#include <vector>
#include <assert.h>
#include "linalglib.h"

namespace linalglib
{
	// Efficient functions for no copying overhead
	template<typename T>
	void add(Vector<T> &x1, Vector<T> &x2, Vector<T> &y)
	{
		int n = x1.size();
		assert(x2.size() == n && y.size() == n);
		for(int i = 0; i < n; i++)
			y[i] = x1[i] + x2[i];		
	}
	template<typename T>
	void subtract(Vector<T> &x1, Vector<T> &x2, Vector<T> &y)
	{
		int n = x1.size();
		assert(x2.size() == n && y.size() == n);
		for(int i = 0; i < n; i++)
			y[i] = x1[i] - x2[i];		
	}

	template<typename T>
	void add(Matrix<T> &M1, Matrix<T> &M2, Matrix<T> &Y)
	{
		int nRows = M1.size();
		assert(nRows > 0 && M2.size() == nRows && Y.size() == nRows);
		int nCols = M1[0].size();
		for(int i = 0; i < nRows; i++)
		{			
			assert(M1[i].size() == nCols && M2[i].size() == nCols && Y[i].size() == nCols);
			for(int j = 0; j < nCols; i++)
				Y[i][j] = M1[i][j] + M2[i][j];
		}
	}
	template<typename T>
	void subtract(Matrix<T> &M1, Matrix<T> &M2, Matrix<T> &Y)
	{
		int nRows = M1.size();
		assert(nRows > 0 && M2.size() == nRows && Y.size() == nRows);
		int nCols = M1[0].size();
		for(int i = 0; i < nRows; i++)
		{			
			assert(M1[i].size() == nCols && M2[i].size() == nCols && Y[i].size() == nCols);
			for(int j = 0; j < nCols; i++)
				Y[i][j] = M1[i][j] - M2[i][j];
		}
	}

	template<typename T>
	void product(Matrix<T> &M, Vector<T> &x, Vector<T> &y)
	{
		int nRows = M.size();
		assert(nRows > 0 && y.size() == nRows);
		int nCols = M[0].size();
		assert(nCols > 0 && x.size() == nCols);
		for(int i = 0; i < nRows; i++)
		{
			T sum = 0;
			assert(M[i].size() == nCols);
			for(int j = 0; j < nCols; j++)			
				sum += M[i][j] * x[j];			
			y[i] = sum;
		}
	}
	template<typename T>
	void dotproduct(Vector<T> &x1, Vector<T> &x2, T &y)
	{
		int n = x1.size();
		assert(x2.size() == n);
		T sum = 0;
		for(int i = 0; i < n; i++)
			sum += x1[i] * x2[i];
		y = sum;
	}
	template<typename T>
	void hadamardproduct(Vector<T> &x1, Vector<T> &x2, Vector<T> &y)
	{
		int n = x1.size();
		assert(x2.size() == n && y.size() == n);
		for(int i = 0; i < n; i++)
			y[i] = x1[i] * x2[i];		
	}
	
	template<typename T>
	void scalarproduct(Vector<T> &x, T &s, Vector<T> &y)
	{
		int n = x.size();
		assert(y.size() == n);
		for(int i = 0; i < n; i++)
			y[i] = x[i] * s;
	}
	template<typename T>
	void scalarproduct(T &s, Vector<T> &x, Vector<T> &y)
	{
		scalarproduct(x, s, y);
	}
	template<typename T>
	void scalarproduct(T &s, Matrix<T> &M, Matrix<T> &Y)
	{
		int nRows = M.size();
		assert(nRows > 0 && Y.size() == nRows);
		int nCols = M[0].size();
		for(int i = 0; i < nRows; i++)
		{			
			assert(M[i].size() == nCols && Y[i].size() == nCols);
			for(int j = 0; j < nCols; i++)
				Y[i][j] = M[i][j] * s;
		}
	}
	template<typename T>
	void scalarproduct(Matrix<T> &M, T &s, Matrix<T> &Y)
	{
		scalarproduct(s, M, Y);
	}

	template<typename T>
	void transpose(Matrix<T> &M, Matrix<T> &Y)
	{
		int nRows_M = M.size();
		int nRows_Y = Y.size();
		assert(nRows_M > 0 && nRows_Y > 0);
		int nCols_M = M[0].size();
		int nCols_Y = Y[0].size();
		assert(nRows_M == nCols_Y && nRows_Y == nCols_M);
		for(int i = 0; i < nRows_M; i++)
			assert(M[i].size() == nCols_M);
		for(int j = 0; j < nRows_Y; j++)
			assert(Y[j].size() == nCols_Y);		
		for(int i = 0; i < nRows_M; i++)
			for(int j = 0; j < nCols_M; j++)
				Y[j][i] = M[i][j];				
	}

	template<typename T>
	void log(Vector<T> &x, Vector<T> &y)
        {
                int n = x.size();
                assert(y.size() == n);
                for(int i = 0; i < x.size(); i++)
                    y[i] = log(x[i]);
        }

	// Overloaded functions with copying overheads
	template<typename T>
	Vector<T> add(Vector<T> &x1, Vector<T> &x2)
	{
		Vector<T> y(x1.size());
		add(x1, x2, y);
		return y;
	}
	template<typename T>
	Vector<T> subtract(Vector<T> &x1, Vector<T> &x2)
	{
		Vector<T> y(x1.size());
		subtract(x1, x2, y);
		return y;
	}

	template<typename T>
	Matrix<T> add(Matrix<T> &M1, Matrix<T> &M2)
	{
		Matrix<T> Y(M1.size(), Vector<T>(M1[0].size()));
		add(M1, M2, Y);
		return Y;
	}
	template<typename T>
	Matrix<T> subtract(Matrix<T> &M1, Matrix<T> &M2)
	{
		Matrix<T> Y(M1.size(), Vector<T>(M1[0].size()));
		subtract(M1, M2, Y);
		return Y;
	}

	template<typename T>
	Vector<T> product(Matrix<T> &M, Vector<T> &x)
	{
		Vector<T> y(M.size());
		product(M, x, y);
		return y;
	}
	template<typename T>
	T dotproduct(Vector<T> &x1, Vector<T> &x2)
	{
		T y;
		dotproduct(x1, x2, y);
		return y;
	}
	template<typename T>
	Vector<T> hadamardproduct(Vector<T> &x1, Vector<T> &x2)
	{
		Vector<T> y(x1.size());
		hadamardproduct(x1, x2, y);
		return y;
	}
	
	template<typename T>
	Vector<T> scalarproduct(Vector<T> &x, T &s)
	{
		Vector<T> y(x.size());
		scalarproduct(x, s, y);
		return y;
	}
	template<typename T>
	Vector<T> scalarproduct(T &s, Vector<T> &x)
	{
		Vector<T> y(x.size());
		scalarproduct(x, s, y);
		return y;	
	}
	template<typename T>
	Matrix<T> scalarproduct(T &s, Matrix<T> &M)
	{
		Matrix<T> Y(M.size(), Vector<T>(M[0].size()));
		scalarproduct(s, M, Y);
		return Y;
	}
	template<typename T>
	Matrix<T> scalarproduct(Matrix<T> &M, T &s)
	{
		Matrix<T> Y(M.size(), Vector<T>(M[0].size()));
		scalarproduct(s, M, Y);
		return Y;	
	}

	template<typename T>
	Matrix<T> transpose(Matrix<T> &M)
	{
		Matrix<T> Y(M[0].size(0), Vector<T>(M.size()));
		transpose(M, Y);
		return Y;
	}
}
