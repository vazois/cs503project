#ifndef IOTOOLS_H
#define IOTOOLS_H

#include "Utils.h"

enum Delimiter{
	COMMA = ',',
	SPACE = ' '
};

enum StructType{
	_1DArrayCudaHost,
	_2DArrayCudaHost,
	_1DArrayHost,
	_2DArrayHost
};

template<typename DATA_T>
class IOTools{
public:
	IOTools():IOTools(0,0,COMMA){};
	IOTools(arr2D dim):IOTools(dim.first,dim.second,COMMA){};

	IOTools(uint64_t rows, uint64_t cols, Delimiter dm){
		this->dim.first = rows;
		this->dim.second = cols;
	};

	~IOTools(){};

	/*Read Data Methods*/
	arr2D fastRead(DATA_T *&arr, std::string file, bool pinned);
	void fastRead(DATA_T *&arr, std::string file, arr2D);
	void fastRead(DATA_T *&arr, std::string file, unsigned int d, unsigned int n);

	/*Write Data Methods*/
	void randDataToFile(unsigned int d, unsigned int n, unsigned int max);
	void randDataToFile(std::string file, unsigned int d, unsigned int n, unsigned int max);
	void randDataToFile(std::string file, unsigned int d, unsigned int n, unsigned int min,unsigned int max);

	/*Utility Methods*/
	arr2D dataDim(std::string);
	uint64_t fsize(std::string);
	inline bool fexists(const std::string& name);
	inline uint64_t cells(arr2D dim);

	void setReadBufferSize(uint64_t readBufferSize){ this->readBufferSize = readBufferSize; }
	void setWriteBufferSize(){ this->writeBufferSize = writeBufferSize; }


private:
	Utils<DATA_T> ut;
	arr2D dim;

	DATA_T *_1DArray;
	DATA_T **_2DArray;

	uint64_t readBufferSize = 16 * 1024;/*in lines*/
	uint64_t writeBufferSize = 128 * 1024; /* in bytes*/
	StructType st;
	Delimiter dm;
};

/*
 * Read Data Methods
 */



/*
 *	Write Data Methods
 */

template<typename DATA_T>
void IOTools<DATA_T>::randDataToFile(unsigned int d, unsigned int c, unsigned int _max){
	std::string file = std::to_string(d) + "_" + std::to_string(c) + ".dat";
	randDataToFile(file,d,c,0,_max);
}

template<typename DATA_T>
void IOTools<DATA_T>::randDataToFile(std::string file, unsigned int d, unsigned int c, unsigned int _max){
	randDataToFile(file,d,c,0,_max);
}

template<typename DATA_T>
void IOTools<DATA_T>::randDataToFile(std::string file, unsigned int d, unsigned int c, unsigned int _min, unsigned int _max){
	std::ofstream fp(file);

	unsigned int lines = MIN(writeBufferSize, c);
	unsigned int total = 0;
	unsigned int processed = 0;
	std::stringstream ss;

	while (processed < c){
		total = processed + lines <= c ? lines : (processed + lines) - c;
		for (unsigned int i = 0; i < total; i++){
			for (unsigned int j = 0; j < d; j++){
				ss << ut.uni(_min, _max);
				if (j < d - 1) ss << ",";
			}
			if (processed + i < c - 1) ss << "\n";
		}

		fp << ss.str();
		ss.clear();
		processed += total;
	}
	fp.close();
}

/*
 * Utility Methods
 */
template<typename DATA_T>
arr2D IOTools<DATA_T>::dataDim(std::string file){
	arr2D dim(0, 0);
	FILE *fp = fopen(file.c_str(), "r");
	char *buffer = new char[readBufferSize];

	while (!feof(fp)){
		unsigned int read = fread(buffer, 1, readBufferSize, fp);
		char *pch = std::strchr(buffer, '\n');
		while (pch != NULL){ dim.second++;  pch = std::strchr(pch + 1, '\n'); }
		memset(buffer, '\0', read);
	}
	dim.second++;
	fclose(fp);

	std::ifstream ifs(file, std::ifstream::in);
	std::string line;
	std::getline(ifs, line);
	dim.first = std::count(line.begin(), line.end(), dm) + 1;

	ifs.close();
	return dim;
}

template<typename DATA_T>
uint64_t IOTools<DATA_T>::fsize(std::string file){
	std::ifstream in(file, std::ifstream::ate | std::ifstream::binary);
	return in.tellg();
}

template<typename DATA_T>
inline bool IOTools<DATA_T>::fexists (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

template<typename DATA_T>
uint64_t IOTools<DATA_T>::cells(arr2D arr){
	return arr.first*arr.second;
}

#endif
