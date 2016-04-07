#include"common/ArgParser.h"
#include"common/Utils.h"

#include"serial/SNeuralNet.h"

#include<iostream>

int main(){
	net_arch arch;

	arch.push_back(2);
	arch.push_back(4);
	arch.push_back(3);
	arch.push_back(1);

	SNeuralNet<float,float,float> nnet(arch);

	nnet.set_batch_size(200);
	nnet.set_error_threshold(0.0000001);
	nnet.set_max_iterations(500);


	nnet.print_configuration();


	vz::pause();
}
