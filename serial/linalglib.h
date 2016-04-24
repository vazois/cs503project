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

	// Matrix, Vector initializers
	template<typename T>
	void zeros(Vector<T> &x);
	template<typename T>
	void zeros(Matrix<T> &M);
	template<typename T>
	void ones(Vector<T> &x);
	template<typename T>
	void ones(Matrix<T> &M);
	template<typename T>
	void constant(Vector<T> &x, T c);
	template<typename T>
	void constant(Matrix<T> &M, T c);

	// Efficient functions for no copying overhead
	template<typename T>
	void add(Vector<T> &x1, Vector<T> &x2, Vector<T> &y);
	template<typename T>
	void sub(Vector<T> &x1, Vector<T> &x2, Vector<T> &y);

	template<typename T>
	void add(Matrix<T> &M1, Matrix<T> &M2, Matrix<T> &Y);
	template<typename T>
	void sub(Matrix<T> &M1, Matrix<T> &M2, Matrix<T> &Y);

	template<typename T>
	void prod(Matrix<T> &M, Vector<T> &x, Vector<T> &y); // matrix vector product
	template<typename T>
	void dprod(Vector<T> &x1, Vector<T> &x2, T &y); // dot product
	template<typename T>
	void hprod(Vector<T> &x1, Vector<T> &x2, Vector<T> &y); // hadamard product
	
	template<typename T>
	void prod(Vector<T> &x, T &s, Vector<T> &y); // product with scalar
	template<typename T>
	void prod(T &s, Vector<T> &x, Vector<T> &y); // product with scalar
	template<typename T>
	void prod(T &s, Matrix<T> &M, Matrix<T> &Y); // product with scalar
	template<typename T>
	void prod(Matrix<T> &M, T &s, Matrix<T> &Y); // product with scalar

	template<typename T>
	void tpose(Matrix<T> &M, Matrix<T> &Y); // transpose

	// Overloaded functions with copying overheads
	template<typename T>
	Vector<T> add(Vector<T> &x1, Vector<T> &x2);
	template<typename T>
	Vector<T> sub(Vector<T> &x1, Vector<T> &x2);

	template<typename T>
	Matrix<T> add(Matrix<T> &M1, Matrix<T> &M2);
	template<typename T>
	Matrix<T> sub(Matrix<T> &M1, Matrix<T> &M2);

	template<typename T>
	Vector<T> prod(Matrix<T> &M, Vector<T> &x); // matrix vector product
	template<typename T>
	T dprod(Vector<T> &x1, Vector<T> &x2); // dot product
	template<typename T>
	Vector<T> hprod(Vector<T> &x1, Vector<T> &x2); // hadamard product
	
	template<typename T>
	Vector<T> prod(Vector<T> &x, T &s); // product with scalar
	template<typename T>
	Vector<T> prod(T &s, Vector<T> &x);	// product with scalar
	template<typename T>
	Matrix<T> prod(T &s, Matrix<T> &M); // product with scalar
	template<typename T>
	Matrix<T> prod(Matrix<T> &M, T &s); // product with scalar

	template<typename T>
	Matrix<T> tpose(Matrix<T> &M); // transpose
}

#endif // LINALGLIB_H