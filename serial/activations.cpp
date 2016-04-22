#include "activations.h"
#include <cmath>
#include <vector>

namespace neuralnet
{	
	template<typename T> 
	inline void Sigmoid<T>::D(T &x, T &y)
	{
		y = F(x)*(1-F(x));
	}

	template<typename T> 
	inline void Sigmoid<T>::F(T &x, T &y)
	{
		y = 1/(1 + exp(-x));
	}

	template<typename T> 
	inline void Sigmoid<T>::operator()(T &x, T &y)
	{
		y = F(x);
	}		

	template<typename T> 
	void Sigmoid<T>::D(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = F(x[i])*(1-F(x[i]));
	}

	template<typename T> 
	void Sigmoid<T>::F(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = 1/(1 + exp(-x[i]));
	}

	template<typename T> 
	void Sigmoid<T>::operator()(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = F(x[i]);
	}

	template<typename T> 
	inline void FSigmoid<T>::D(T &x, T &y)
	{
		y = 1/pow(1 + std::abs(x),2);
	}

	template<typename T> 
	inline void FSigmoid<T>::F(T &x, T &y)
	{
		y = x/(1 + std::abs(x));
	}

	template<typename T> 
	inline void FSigmoid<T>::operator()(T &x, T &y)
	{
		y = F(x);
	}		

	template<typename T> 
	void FSigmoid<T>::D(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = 1/pow(1 + std::abs(x[i]),2);
	}

	template<typename T> 
	void FSigmoid<T>::F(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = x[i]/(1 + std::abs(x[i]));
	}

	template<typename T> 
	void FSigmoid<T>::operator()(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = F(x[i]);
	}

	template<typename T> 
	inline void ReLU<T>::D(T &x, T &y)
	{
		y = (x >= 0 ? 1 : 0);
	}

	template<typename T> 
	inline void ReLU<T>::F(T &x, T &y)
	{
		y = (x >= 0 ? x : 0);
	}

	template<typename T> 
	inline void ReLU<T>::operator()(T &x, T &y)
	{
		y = F(x);
	}		

	template<typename T> 
	void ReLU<T>::D(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = (x[i] >= 0 ? 1 : 0);
	}

	template<typename T> 
	void ReLU<T>::F(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = (x >= 0 ? x[i] : 0);
	}

	template<typename T> 
	void ReLU<T>::operator()(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = F(x[i]);
	}

}