#include <stdio.h>
#include <stdlib.h>

void MatrixInit(float *M, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            M[i * p + j] = (static_cast<float>(rand()) / RAND_MAX) * 2 - 1;
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


int CPU_test() {
    //MatrixAdd test
    int n = 3, p = 3;
    float *M1, *M2, *Mout;
    M1 = (float *)malloc(n * p * sizeof(float));
    M2 = (float *)malloc(n * p * sizeof(float));

    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

    Mout = (float *)malloc(n * p * sizeof(float));
    MatrixAdd(M1, M2, Mout, n, p);

    //MatrixPrint(M1, n, p);
    //MatrixPrint(M2, n, p);
    //MatrixPrint(Mout, n, p);

    free(M1);
    free(M2);
    free(Mout);

    //MatrixMult test
    n = 2;
    float *M3, *M4, *Mout2;
    M3 = (float *)malloc(n * n * sizeof(float));
    M4 = (float *)malloc(n * n * sizeof(float));

    MatrixInit(M3, n, n);
    MatrixInit(M4, n, n);

    Mout2 = (float *)malloc(n * n * sizeof(float));
    MatrixMult(M3, M4, Mout2, n);

    MatrixPrint(M3, n, n);
    MatrixPrint(M4, n, n);
    MatrixPrint(Mout2, n, n);

    free(M3);
    free(M4);
    free(Mout2);

    return 0;
}

int GPUtest() {
    //cudaMatrixAdd test
    int n = 3, p = 3;
    float *M1, *M2, *Mout;
    float *d_M1, *d_M2, *d_Mout;

    M1 = (float *)malloc(n * p * sizeof(float));
    M2 = (float *)malloc(n * p * sizeof(float));

    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

    Mout = (float *)malloc(n * p * sizeof(float));

    cudaMalloc((void **)&d_M1, n * p * sizeof(float));
    cudaMalloc((void **)&d_M2, n * p * sizeof(float));
    cudaMalloc((void **)&d_Mout, n * p * sizeof(float));

    cudaMemcpy(d_M1, M1, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, n * p * sizeof(float), cudaMemcpyHostToDevice);

    cudaMatrixAdd<<<n, p>>>(d_M1, d_M2, d_Mout, n, p);

    cudaMemcpy(Mout, d_Mout, n * p * sizeof(float), cudaMemcpyDeviceToHost);

    //MatrixPrint(M1, n, p);
    //MatrixPrint(M2, n, p);

    //MatrixPrint(Mout, n, p);

    free(M1);
    free(M2);
    free(Mout);

    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);



    // cudaMatrixMult test

    n = 2;
    float *M3, *M4, *Mout2;
    float *d_M3, *d_M4, *d_Mout2;

    M3 = (float *)malloc(n * n * sizeof(float));
    M4 = (float *)malloc(n * n * sizeof(float));

    MatrixInit(M3, n, n);
    MatrixInit(M4, n, n);

    Mout2 = (float *)malloc(n * n * sizeof(float));

    cudaMalloc((void **)&d_M3, n * n * sizeof(float));
    cudaMalloc((void **)&d_M4, n * n * sizeof(float));
    cudaMalloc((void **)&d_Mout2, n * n * sizeof(float));

    cudaMemcpy(d_M3, M3, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M4, M4, n * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMatrixMult<<<n, n>>>(d_M3, d_M4, d_Mout2, n);

    cudaMemcpy(Mout2, d_Mout2, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    MatrixPrint(M3, n, n);
    MatrixPrint(M4, n, n);

    MatrixPrint(Mout2, n, n);

    free(M3);
    free(M4);
    free(Mout2);

    cudaFree(d_M3);
    cudaFree(d_M4);
    cudaFree(d_Mout2);

    return 0;

}

int main() {
    
    //CPU_test();
    GPUtest();
    return 0;
}
