#ifndef UTILS_H
#define UTILS_H

#define DEBUG false
#define TEST false

#define toDigit(c) c - '0'

#include <iostream>
#include <errno.h>
#include <random>
#include <time.h>
#include <vector>
#include <iterator>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <algorithm>

#include "CudaHelper.h"
#include "Time.h"

#define PI 3.14159265

#define TESTB(v,p) ((v) & (1<<(p)))
#define SETB(X) (1 << X)

#define MIN(A,B) (A < B ? A : B)
#define MAX(A,B) (A > B ? A : B)

#define BOOL(B) (B == 0 ? "false" : "true")

#define FOLDER "data/"

namespace vz{
	static void pause(){ std::cout << "\nPress enter to continue ..." << std::endl; while (getchar() != '\n'); }
	static void error(std::string error){
		std::cout << "RunTime Error Encountered:("<<error<<")\nPress enter to continue ..." << std::endl; while (getchar() != '\n');
		exit(1);
	}
}

template<class T>
class Utils{
public:
	Utils(){
		srand(time(0)*rand());
		this->seed = PI*(rand() % INT_MAX);
		this->generator.seed(this->seed);
	};
	~Utils(){};

	//Random Number Generators//
	T uni(T max);
	T uni(T min, T max);
	T rnum(unsigned int low, unsigned int high);
	void setSeed(unsigned int);
	void randDataToFile(unsigned int d, unsigned int n, unsigned int max);
	void randDataToFile(std::string file, unsigned int d, unsigned int n, unsigned int max);
	void randDataToFile(std::string file, unsigned int d, unsigned int n, unsigned int min,unsigned int max);
	void shuffle_array(T *&arr, unsigned int n);

	//String Tokenize//
	std::vector<std::string> split(std::string str, std::string delimiter);

	//File Manipulation Files//
	arr2D dataDim(std::string);//Default delimiter ","
	arr2D dataDim(std::string, std::string);//choose delimeter to find dimensions of data
	uint64_t fsize(std::string);
	inline bool fexists(const std::string& name);

	arr2D fastRead(T *&arr, std::string file, bool pinned);
	void fastRead(T *&arr, std::string file, arr2D);
	void fastRead(T *&arr, std::string file, unsigned int d, unsigned int n);
	void mt_fastRead(T *&arr, std::string file, unsigned int d, unsigned int n);

	void print_array(const T *arr, unsigned int limit);

protected:
	unsigned int seed;
	std::string delim = ",";	
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution{ 0.0, 1.0 };
};

/*

Method and Construstor implementations

*/

static uint64_t cells(arr2D arr){
	return arr.first*arr.second;
}

static void print(arr2D arr){
	std::cout << "("<<arr.first<<","<<arr.second<<")" << std::endl;
}

/***********************/
/*	  Utils Class	   */
/***********************/

template<class T>
T Utils<T>::uni(T max){
	return this->distribution(generator)*max;
}

template<class T>
T Utils<T>::uni(T min, T max){
	return this->distribution(generator)*max + min;
}

template<class T>
T Utils<T>::rnum(unsigned int low, unsigned high){
	srand(time(0)*rand()); return PI*(rand() % high + low);
}

template<class T>
void Utils<T>::setSeed(unsigned int seed){
	this->seed = seed;
	this->generator.seed(this->seed);
}

template<class T>
void Utils<T>::randDataToFile(unsigned int d, unsigned int n, unsigned int max){
	std::string file = std::to_string(d) + "_" + std::to_string(n) + ".dat";
	this->randDataToFile(file, d, n, 0, max);
}

template<class T>
void Utils<T>::randDataToFile(std::string file, unsigned int d, unsigned int n, unsigned int max){
	this->randDataToFile(file, d, n, 0, max);
}

template<class T>
void Utils<T>::randDataToFile(std::string file, unsigned int d, unsigned int n, unsigned int _min, unsigned int _max){
	std::ofstream fp(file);

	unsigned int lines = MIN(4096, n);
	unsigned int total = 0;
	unsigned int processed = 0;
	std::stringstream ss;

	while (processed < n){
		total = processed + lines <= n ? lines : (processed + lines) - n;
		for (unsigned int i = 0; i < total; i++){
			for (unsigned int j = 0; j < d; j++){
				ss << std::setprecision(2) << this->uni(_min, _max);
				if (j < d - 1) ss << ",";
			}
			if (processed + i < n - 1) ss << "\n";
		}

		fp << ss.str();
		ss.clear();
		processed += total;
	}
	fp.close();
}

template<class T>
void Utils<T>::shuffle_array(T *&arr,unsigned int n){
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	shuffle(arr, (arr+n), std::default_random_engine(seed));
}

template<class T>
arr2D Utils<T>::dataDim(std::string file, std::string delim){
	this->delim = delim;
	return this->dataDim(file);
}

template<class T>
arr2D Utils<T>::dataDim(std::string file){
	arr2D dim(0, 0);
	FILE *fp = fopen(file.c_str(), "r");
	uint64_t bytes = 1024;
	char *buffer = new char[bytes];
	while (!feof(fp)){
		unsigned int read = fread(buffer, 1, bytes, fp);
		char *pch = std::strchr(buffer, '\n');
		while (pch != NULL){ dim.second++;  pch = std::strchr(pch + 1, '\n'); }
		memset(buffer, '\0', read);
	}
	dim.second++;
	fclose(fp);

	std::ifstream ifs(file, std::ifstream::in);
	std::string line;
	std::getline(ifs, line);
	dim.first = std::count(line.begin(), line.end(), ',') + 1;

	ifs.close();
	return dim;
}

template<class T>
inline bool Utils<T>::fexists (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

template<class T>
std::vector < std::string > Utils<T>::split(std::string str, std::string delimiter){
	std::vector<std::string> out;
	std::string lstr(str);
	int pos = 0;

	while ((pos = lstr.find(delimiter)) != -1){
		out.push_back(lstr.substr(0, pos));
		lstr = lstr.substr(pos + 1);
	}
	out.push_back(lstr);
	return out;
}

template<class T>
uint64_t Utils<T>::fsize(std::string file){
	std::ifstream in(file, std::ifstream::ate | std::ifstream::binary);
	return in.tellg();
}

template<class T>
arr2D Utils<T>::fastRead(T *&arr, std::string file, bool pinned){
	arr2D dim = this->dataDim(file);
	if(pinned) allocHostMem<T>(&arr,sizeof(T)*dim.first*dim.second,"Error Allocating Host Pinned Memory (fastRead)");
	else arr = new T[dim.first*dim.second];
	fastRead(arr,file,dim);
	return dim;
}

template<class T>
void Utils<T>::fastRead(T *&arr, std::string file, arr2D dim){
	this->fastRead(arr, file, dim.first, dim.second);
}

/**/
template<class T>
void Utils<T>::fastRead(T *&arr, std::string file, unsigned int d, unsigned int n){
	if(arr == NULL) vz::error("fastRead: Array NULL pointer exception");
	if(!this->fexists(file)) vz::error("fastRead: File Not Found Exception");
	uint64_t totalbytes = this->fsize(file);
	char *buffer = new char[totalbytes];
	FILE *fp = fopen(file.c_str(), "r");
	totalbytes = fread(buffer, 1, totalbytes, fp);
	buffer[totalbytes] = '\0';
	fclose(fp);

	std::stringstream iss; iss << buffer;
	delete buffer;

	int i = 0;
	char dummy;
	while(i < d*n){ iss>>arr[i]; iss >> dummy; i++; }
}

template<class T>
void Utils<T>::print_array(const T *arr,unsigned int limit){
	for(int i =0;i<limit;i++) std::cout<<arr[i] << " ";
	std::cout<<std::endl;
}

#endif
