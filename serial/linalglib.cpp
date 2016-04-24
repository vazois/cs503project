#include <assert.h>
#include "linalglib.h"

namespace linalglib
{
	// Matrix, Vector initializers
	template<typename T>
	void zeros(Vector<T> &x)
	{
		constant(x, 0);
	}
	template<typename T>
	void zeros(Matrix<T> &M)
	{
		constant(M, 0);
	}
	template<typename T>
	void ones(Vector<T> &x)
	{
		constant(x, 1);
	}
	template<typename T>
	void ones(Matrix<T> &M)
	{
		constant(M, 1);
	}
	template<typename T>
	void constant(Vector<T> &x, T c)
	{	
		int n = x.size();
		for(int i = 0; i < n; i++)
			x[i] = c;
	}
	template<typename T>
	void constant(Matrix<T> &M, T c)
	{	
		int nRows = M.size();
		assert(nRows > 0);
		int nCols = M[0].size();
		for(int i = 0; i < nRows; i++)
		{
			assert(M[i].size() == nCols);
			for(int j = 0; j < nCols; j++)
			{
				M[i][j] = c;
			}
		}
	}

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
	void sub(Vector<T> &x1, Vector<T> &x2, Vector<T> &y)
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
	void sub(Matrix<T> &M1, Matrix<T> &M2, Matrix<T> &Y)
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
	void prod(Matrix<T> &M, Vector<T> &x, Vector<T> &y)
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
	void dprod(Vector<T> &x1, Vector<T> &x2, T &y)
	{
		int n = x1.size();
		assert(x2.size() == n);
		T sum = 0;
		for(int i = 0; i < n; i++)
			sum += x1[i] * x2[i];
		y = sum;
	}
	template<typename T>
	void hprod(Vector<T> &x1, Vector<T> &x2, Vector<T> &y)
	{
		int n = x1.size();
		assert(x2.size() == n && y.size() == n);
		for(int i = 0; i < n; i++)
			y[i] = x1[i] * x2[i];		
	}
	
	template<typename T>
	void prod(Vector<T> &x, T &s, Vector<T> &y)
	{
		int n = x.size();
		assert(y.size() == n);
		for(int i = 0; i < n; i++)
			y[i] = x[i] * s;
	}
	template<typename T>
	void prod(T &s, Vector<T> &x, Vector<T> &y)
	{
		prod(x, s, y);
	}
	template<typename T>
	void prod(T &s, Matrix<T> &M, Matrix<T> &Y)
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
	void prod(Matrix<T> &M, T &s, Matrix<T> &Y)
	{
		prod(s, M, Y);
	}

	template<typename T>
	void tpose(Matrix<T> &M, Matrix<T> &Y)
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
	Vector<T> sub(Vector<T> &x1, Vector<T> &x2)
	{
		Vector<T> y(x1.size());
		sub(x1, x2, y);
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
	Matrix<T> sub(Matrix<T> &M1, Matrix<T> &M2)
	{
		Matrix<T> Y(M1.size(), Vector<T>(M1[0].size()));
		sub(M1, M2, Y);
		return Y;
	}

	template<typename T>
	Vector<T> prod(Matrix<T> &M, Vector<T> &x)
	{
		Vector<T> y(M.size());
		prod(M, x, y);
		return y;
	}
	template<typename T>
	T dprod(Vector<T> &x1, Vector<T> &x2)
	{
		T y;
		dprod(x1, x2, y);
		return y;
	}
	template<typename T>
	Vector<T> hprod(Vector<T> &x1, Vector<T> &x2)
	{
		Vector<T> y(x1.size());
		hprod(x1, x2, y);
		return y;
	}
	
	template<typename T>
	Vector<T> prod(Vector<T> &x, T &s)
	{
		Vector<T> y(x.size());
		prod(x, s, y);
		return y;
	}
	template<typename T>
	Vector<T> prod(T &s, Vector<T> &x)
	{
		Vector<T> y(x.size());
		prod(x, s, y);
		return y;	
	}
	template<typename T>
	Matrix<T> prod(T &s, Matrix<T> &M)
	{
		Matrix<T> Y(M.size(), Vector<T>(M[0].size()));
		prod(s, M, Y);
		return Y;
	}
	template<typename T>
	Matrix<T> prod(Matrix<T> &M, T &s)
	{
		Matrix<T> Y(M.size(), Vector<T>(M[0].size()));
		prod(s, M, Y);
		return Y;	
	}

	template<typename T>
	Matrix<T> tpose(Matrix<T> &M)
	{
		Matrix<T> Y(M[0].size(0), Vector<T>(M.size()));
		tpose(M, Y);
		return Y;
	}
}
