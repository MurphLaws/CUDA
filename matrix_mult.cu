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

int main() {
    float **M;
    int n = 3, p = 4;

    // Allocate memory for the 2D array
    M = (float **)malloc(n * sizeof(float *));
    for (int i = 0; i < n; ++i) {
        M[i] = (float *)malloc(p * sizeof(float));
    }

    // Initialize the matrix
    MatrixInit(M, n, p);

    // Print the matrix
    MatrixPrint(M, n, p);

    // Free the allocated memory
    for (int i = 0; i < n; ++i) {
        free(M[i]);
    }
    free(M);

    return 0;
}
