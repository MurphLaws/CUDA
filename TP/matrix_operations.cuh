// matrix_operations.cuh
#ifndef MATRIX_MULT_CUH
#define MATRIX_MULT_CUH

void MatrixInit(float *M, int n, int p);
void MatrixPrint(float *M, int n, int p);
void MatrixAdd(float *A, float *B, float *C, int n, int p);
void MatrixInit01(float *M, int n, int p);
void MatrixInit3D(float *M, int n, int p, int q);
void MatrixMult(float *A, float *B, float *C, int n, int p, int m);
__global__ void cudaMatrixAdd(float *A, float *B, float *C, int n, int p);
__global__ void cudaMatrixMult(float *A, float *B, float *C, int n, int p, int m);


#endif // MATRIX_MULT_CUH