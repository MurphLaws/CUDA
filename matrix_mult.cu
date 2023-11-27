#include <stdio.h>
#include <stdlib.h>

void MatrixInit(float **M, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            M[i][j] = (static_cast<float>(rand()) / RAND_MAX) * 2 - 1;
}

void MatrixPrint(float **M, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++) {
        printf("\n");
        for (j = 0; j < p; j++)
            printf("%f ", M[i][j]);
    }
    printf("\n");
}

void MatrixAdd(float **M1, float **M2, float **Mout, int n, int p) {
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            Mout[i][j] = M1[i][j] + M2[i][j];
}



int main() {
    float **M, **M1, **M2, **Mout;
    int n = 3, p = 4;

    M = (float **)malloc(n * sizeof(float *));
    M1 = (float **)malloc(n * sizeof(float *));
    M2 = (float **)malloc(n * sizeof(float *));
    Mout = (float **)malloc(n * sizeof(float *));

    for (int i = 0; i < n; i++) {
        M[i] = (float *)malloc(p * sizeof(float));
        M1[i] = (float *)malloc(p * sizeof(float));
        M2[i] = (float *)malloc(p * sizeof(float));
        Mout[i] = (float *)malloc(p * sizeof(float));
    }

    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

    MatrixAdd(M1, M2, Mout, n, p);

    MatrixPrint(M1, n, p);
    MatrixPrint(M2, n, p);
    MatrixPrint(Mout, n, p);

 

    for (int i = 0; i < n; i++) {
        free(M[i]);
        free(M1[i]);
        free(M2[i]);
        free(Mout[i]);
    }

    return 0;
}
