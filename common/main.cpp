#include "ArgParser.h"

#include "Time.h"
//#include "CudaHelper.h"
#include "Utils.h"

/*Functor Example*/
template<typename T>
T myFunc(T x , T y){
	return x*y + y;
}

template<typename T,typename FuncType>
T doMath (T x, T y, FuncType func)
{
    return func( x, y );
}

void test (){
	float a  = 0.12 , b = 0.12;
	float result = doMath<float>(a,b,myFunc<float>);
	std::cout<<"result<>: " << result << std::endl;
}

struct Default{
	template<typename T>
	inline T operator()(T x){
		return x;
	}
};

struct Sigmoid{
	template<typename T>
	inline T operator()(T x){
		return 1/(1 + exp(-x));
	}
};

struct FSigmoid{
	template<typename T>
	inline T operator()(T x){
		return x/(1.0 + fabs(x));
	}
};

template<typename T,typename ACT>
struct Layer{
	ACT F;
	T x;

	//Layer(ACT F){
	Layer(ACT F){
		this->F = F;
		x = 1.2;
	}

	void print(){
		std::cout<<x<<std::endl;
	}

	T compute(){
		return F(1.2);
	}

};


int main(int argc, char **argv){
	Default d;
	Sigmoid s;
	FSigmoid fs;
	
	std::cout<<d('x')<< std::endl;
	std::cout<<s(1.2)<< std::endl;
	std::cout<<fs(1.2)<< std::endl;

	Layer<float,Default> ld(d);
	Layer<float,Sigmoid> ls(s);
	Layer<float,FSigmoid> lfs(fs);

	float x = 1.2;
	std::cout<<"----\n";
	std::cout<<ld.compute()<< std::endl;
	std::cout<<ls.compute()<< std::endl;
	std::cout<<lfs.compute()<< std::endl;

	//d=s;

	//std::cout<<d(1.2)<<std::endl;

	vz::pause();

	return 0;
}
