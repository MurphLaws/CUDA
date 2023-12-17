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



void MatrixInit01(float *M, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            M[i * p + j] = (static_cast<float>(rand()) / RAND_MAX);
}

//Declare the method MatrixInitFilter that initialize a 3d matric of size n*p*q with 0 values

void MatrixInit3D(float *M, int n, int p, int q) {
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            for (k = 0; k < q; k++)
                M[i * p * q + j * q + k] = 0;
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



