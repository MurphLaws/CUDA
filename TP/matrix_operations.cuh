// matrix_operations.cuh
#ifndef MATRIX_MULT_CUH
#define MATRIX_MULT_CUH

void MatrixInit(float *M, int n, int p);
void MatrixPrint(float *M, int n, int p);
void MatrixAdd(float *A, float *B, float *C, int n, int p);
void MatrixInit01_2D(float *M, int n, int p);
void MatrixInit0_3D(float *M, int n, int p, int q);
void MatrixInit01_3D(float *M, int n, int p, int q);
void MatrixMult(float *A, float *B, float *C, int n, int p, int m);
void MatrixPrint3D(float *M, int n, int p, int q);
__global__ void subsampling(float *input, int input_size, int kernel_size, float *output);
__global__ void convolution(float *input, int input_size, int kernel_size, int number_of_filters, float *kernel, float *output);
__global__ void cudaMatrixSum(float *M, float *sum, int n, int p);
__global__ void cudaMatrixAdd(float *A, float *B, float *C, int n, int p);
__global__ void cudaMatrixMult(float *A, float *B, float *C, int n, int p, int m);


#endif // MATRIX_MULT_CUH