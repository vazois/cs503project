#include "ArgParser.h"

#include "Time.h"
//#include "CudaHelper.h"
#include "Utils.h"

/*Functor Example*/
float myFunc(float x , float y){
	return x*y + y;
}

template<typename FuncType>
float doMath (float x, float y, FuncType func)
{
    return func( x, y );
}

void test (){
	float a  = 0.32 , b = 0.12;
	float result = doMath(a,b,myFunc);
	std::cout<<"result: " << result << std::endl;
}
/**/


int main(int argc, char **argv){
	ArgParser ap;
	
	ap.parseArgs(argc,argv);

	if(ap.exists(HELP) || ap.count() == 0){
		ap.menu();
		return 0;
	}
	test();
	vz::pause();

	return 0;
}
