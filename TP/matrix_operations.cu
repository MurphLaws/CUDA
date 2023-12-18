#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix_operations.cuh"

/*
L'architecture du réseau LeNet-5 est composé de plusieurs couches :

Layer 1- Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de donnée MNIST

Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.

Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.

*/

void MatrixInit(float *M, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            M[i * p + j] = (static_cast<float>(rand()) / RAND_MAX) * 2 - 1;
}



void MatrixInit01_2D(float *M, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            M[i * p + j] = (static_cast<float>(rand()) / RAND_MAX);
}

//Declare the method MatrixInitFilter that initialize a 3d matric of size n*p*q with 0 values

void MatrixInit0_3D(float *M, int n, int p, int q) {
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            for (k = 0; k < q; k++)
                M[i * p * q + j * q + k] = 0;
}


//Declare the method MatrixInit01_3D that initialize a 3d matric of size n*p*q with random valñues between 0 and 1

void MatrixInit01_3D(float *M, int n, int p, int q) {
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            for (k = 0; k < q; k++)
                M[i * p * q + j * q + k] = 1;//(static_cast<float>(rand()) / RAND_MAX);
}


void MatrixPrint(float *M, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++) {
        printf("\n");
        for (j = 0; j < p; j++)
            printf("%f\t", M[i * p + j]);
    
    }
    printf("\n");
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];


}


void MatrixMult(float *M1, float *M2, float *Mout, int n){
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            Mout[i * n + j] = 0;
            for (k = 0; k < n; k++)
                Mout[i * n + j] += M1[i * n + k] * M2[k * n + j];
        }
    
}


//Define a print 3d matrix method that prints a 3d matrix of size n*p*q, printing each matrix of size p*q separated 
// by a line


void MatrixPrint3D(float *M, int n, int p, int q) {
    int i, j, k;
    for (i = 0; i < n; i++) {
        printf("\n");
        for (j = 0; j < p; j++) {
            printf("\n");
            for (k = 0; k < q; k++)
                printf("%f\t", M[i * p * q + j * q + k]);
        }
        printf("\n");
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k;
    Mout[i * n + j] = 0;
    for (k = 0; k < n; k++)
        Mout[i * n + j] += M1[i * n + k] * M2[k * n + j];
}

//Define a method that sum all the elements into a matrix

__global__ void cudaMatrixSum(float *M, float *sum, int n, int p) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    sum[i * p + j] = 0;
    for (int k = 0; k < n; k++)
        sum[i * p + j] += M[i * p + k];
}




__global__ void convolution(float *input, int input_size, int kernel_size, int number_of_filters, float *kernel, float *output) {
    int output_size = input_size - kernel_size + 1;
    int i = threadIdx.x; // Current row index
    int j = threadIdx.y; // Current column index

    // Iterate over filters
    for (int f = 0; f < number_of_filters; f++) {
        // Iterate over output positions
        for (int k = 0; k < output_size; k++) {
            for (int l = 0; l < output_size; l++) {
                float sum = 0;

                // Perform convolution operation
                for (int m = 0; m < kernel_size; m++) {
                    for (int n = 0; n < kernel_size; n++) {
                        int input_offset = (i + k + m) * input_size + (j + l + n);
                        int kernel_offset = f * kernel_size * kernel_size + m * kernel_size + n;
                        sum += input[input_offset] * kernel[kernel_offset];
                    }
                }

                int output_offset = f * output_size * output_size + k * output_size + l;
                output[output_offset] = sum;
            }
        }
    }
}



__global__ void subsampling(float *input, int input_size, int kernel_size, int number_of_filters, float *output) {
    int output_size = input_size / kernel_size;
    int i = threadIdx.x; // Current row index
    int j = threadIdx.y; // Current column index
    int f = blockIdx.x; // Current filter index

    // Iterate over output positions
    for (int k = 0; k < output_size; k++) {
        for (int l = 0; l < output_size; l++) {
            float sum = 0;

            // Perform subsampling operation
            for (int m = 0; m < kernel_size; m++) {
                for (int n = 0; n < kernel_size; n++) {
                    int input_offset = (f * input_size + (i + k * kernel_size + m) * input_size + (j + l * kernel_size + n));
                    sum += input[input_offset];
                }
            }

            int output_offset = (f * output_size + i) * output_size + j;
            output[output_offset] = sum / (kernel_size * kernel_size);
        }
    }
}