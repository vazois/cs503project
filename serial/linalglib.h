#ifndef LINALGLIB_H
#define LINALGLIB_H

#include <vector>

namespace linalglib
{
	// Typedefs
	template<typename T>
	using Vector = std::vector<T>;
	template<typename T>
	using Matrix = Vector<Vector<T> >;

	// Efficient functions for no copying overhead
	template<typename T>
	void add(Vector<T> &x1, Vector<T> &x2, Vector<T> &y);
	template<typename T>
	void subtract(Vector<T> &x1, Vector<T> &x2, Vector<T> &y);

	template<typename T>
	void add(Matrix<T> &M1, Matrix<T> &M2, Matrix<T> &Y);
	template<typename T>
	void subtract(Matrix<T> &M1, Matrix<T> &M2, Matrix<T> &Y);

	template<typename T>
	void product(Matrix<T> &M, Vector<T> &x, Vector<T> &y);
	template<typename T>
	void dotproduct(Vector<T> &x1, Vector<T> &x2, T &y);
	template<typename T>
	void hadamardproduct(Vector<T> &x1, Vector<T> &x2, Vector<T> &y);
	
	template<typename T>
	void scalarproduct(Vector<T> &x, T &s, Vector<T> &y);
	template<typename T>
	void scalarproduct(T &s, Vector<T> &x, Vector<T> &y);	
	template<typename T>
	void scalarproduct(T &s, Matrix<T> &M, Matrix<T> &Y);
	template<typename T>
	void scalarproduct(Matrix<T> &M, T &s, Matrix<T> &Y);

	template<typename T>
	void transpose(Matrix<T> &M, Matrix<T> &Y);

	// Overloaded functions with copying overheads
	template<typename T>
	Vector<T> add(Vector<T> &x1, Vector<T> &x2);
	template<typename T>
	Vector<T> subtract(Vector<T> &x1, Vector<T> &x2);

	template<typename T>
	Matrix<T> add(Matrix<T> &M1, Matrix<T> &M2);
	template<typename T>
	Matrix<T> subtract(Matrix<T> &M1, Matrix<T> &M2);

	template<typename T>
	Vector<T> product(Matrix<T> &M, Vector<T> &x);
	template<typename T>
	T dotproduct(Vector<T> &x1, Vector<T> &x2);
	template<typename T>
	Vector<T> hadamardproduct(Vector<T> &x1, Vector<T> &x2);
	
	template<typename T>
	Vector<T> scalarproduct(Vector<T> &x, T &s);
	template<typename T>
	Vector<T> scalarproduct(T &s, Vector<T> &x);	
	template<typename T>
	Matrix<T> scalarproduct(T &s, Matrix<T> &M);
	template<typename T>
	Matrix<T> scalarproduct(Matrix<T> &M, T &s);

	template<typename T>
	Matrix<T> transpose(Matrix<T> &M);
}

#endif // LINALGLIB_H