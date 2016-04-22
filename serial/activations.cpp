#include "layer.h"
#include <cmath>
#include <vector>

namespace neuralnet
{	
	template<typename T> 
	inline void Sigmoid::D(T &x, T &y)
	{
		y = F(x)*(1-F(x));
	}

	template<typename T> 
	inline void Sigmoid::F(T &x, T &y)
	{
		y = 1/(1 + exp(-x));
	}

	template<typename T> 
	inline void Sigmoid::operator()(T &x, T &y)
	{
		y = F(x);
	}		

	template<typename T> 
	void Sigmoid::D(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = F(x[i])*(1-F(x[i]));
	}

	template<typename T> 
	void Sigmoid::F(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = 1/(1 + exp(-x[i]));
	}

	template<typename T> 
	void Sigmoid::operator()(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = F(x[i]);
	}

	template<typename T> 
	inline void FSigmoid::D(T &x, T &y)
	{
		y = 1/pow(1 + std::abs(x),2);
	}

	template<typename T> 
	inline void FSigmoid::F(T &x, T &y)
	{
		y = x/(1 + std::abs(x));
	}

	template<typename T> 
	inline void FSigmoid::operator()(T &x, T &y)
	{
		y = F(x);
	}		

	template<typename T> 
	void FSigmoid::D(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = 1/pow(1 + std::abs(x[i]),2);
	}

	template<typename T> 
	void FSigmoid::F(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = x[i]/(1 + std::abs(x[i]));
	}

	template<typename T> 
	void FSigmoid::operator()(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = F(x[i]);
	}

	template<typename T> 
	inline void ReLU::D(T &x, T &y)
	{
		y = (x >= 0 ? 1 : 0);
	}

	template<typename T> 
	inline void ReLU::F(T &x, T &y)
	{
		y = (x >= 0 ? x : 0);
	}

	template<typename T> 
	inline void ReLU::operator()(T &x, T &y)
	{
		y = F(x);
	}		

	template<typename T> 
	void ReLU::D(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = (x[i] >= 0 ? 1 : 0);
	}

	template<typename T> 
	void ReLU::F(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = (x >= 0 ? x[i] : 0);
	}

	template<typename T> 
	void ReLU::operator()(std::vector<T> &x, std::vector<T> &y)
	{
		for(int i = 0; i < x.size(); i++)
			y[i] = F(x[i]);
	}

}