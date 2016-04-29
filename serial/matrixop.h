#ifndef MATRIXOP_H
#define MATRIXOP_H

void prod(float *x, float s, float *y, int n);

void prod(float s, float *x, float *y, int n);

void mvProd(float **M, float *x, float *y, int m, int n);

void mvProdT(float **M, float *x, float *y, int m, int n);

void add(float *x1, float *x2, float* y, int n);

void hprod(float *x1, float *x2, float *y, int n);

void sigmoid(float *x, float *y, int n);

void dSigmoid(float *x, float *y, int n);

void softmax(float *x, float *y, int n);

void softmaxD(float *x, float *y, int n);

float costFn(float *y_train, float *y_pred, float ***w, float lambda, int m, int n, int l, int n_out);

void costFnD(float *y_train, float *y_pred, float *delC_a, int n);

bool equals(float* pred, float* label, int n);

#endif // MATRIXOP_H