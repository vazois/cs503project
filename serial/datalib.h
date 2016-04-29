#ifndef DATALIB_H
#define DATALIB_H

const int NUM_CLASSES = 10;
const int NUM_TRAIN = 50000;
const int NUM_TEST = 10000;
const int NUM_VAL = 10000;
const int NUM_FEATURES = 784;

float **x_train, **y_train, **x_test, **y_test, **x_val, **y_val;

void readData();

#endif // DATALIB_H