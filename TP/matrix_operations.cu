#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix_operations.cuh"


#define WIDTH 28
#define HEIGHT 28


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
void MatrixInit0_3D(float *M, int n, int p, int q) {
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            for (k = 0; k < q; k++)
                M[i * p * q + j * q + k] = 0;
}



void MatrixInit01_3D(float *M, int n, int p, int q) {
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            for (k = 0; k < q; k++)
                M[i * p * q + j * q + k] = (static_cast<float>(rand()) / RAND_MAX);
}

void MatrixInit0(float *M, int n) {
    int i;
    for (i = 0; i < n; i++)
        M[i] = 0;
}


void MatrixInitFilter(float *M, int n, int p, int q) {
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j < p; j++)
            for (k = 0; k < q; k++)
                M[i * p * q + j * q + k] = (static_cast<float>(rand()) / RAND_MAX) * 2 - 1;
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


__global__ void cudaMatrixSum(float *M, float *sum, int n, int p) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    sum[i * p + j] = 0;
    for (int k = 0; k < n; k++)
        sum[i * p + j] += M[i * p + k];
}




__global__ void convolution(float *input, int input_size, int kernel_size, int number_of_filters, float *kernel, float *output) {
    int output_size = input_size - kernel_size + 1;
    int i = threadIdx.x;
    int j = threadIdx.y; 
 

    for (int f = 0; f < number_of_filters; f++) {
      
        for (int k = 0; k < output_size; k++) {
            for (int l = 0; l < output_size; l++) {
                float sum = 0;

                
                for (int m = 0; m < kernel_size; m++) {
                    for (int n = 0; n < kernel_size; n++) {
                        int input_offset = (i + k + m) * input_size + (j + l + n);
                        int kernel_offset = f * kernel_size * kernel_size + m * kernel_size + n;
                        sum += input[input_offset] * kernel[kernel_offset];
                    }
                }

                int output_offset = f * output_size * output_size + k * output_size + l;
                output[output_offset] = tanhf(sum);
            }
        }
    }}


__global__ void avgPooling(float *M, float *P, int n, int p, int q, int poolSize) {
    int i = blockIdx.x;
    int j = threadIdx.y * poolSize;
    int k = threadIdx.x * poolSize;

    if (i < n && j < p && k < q) {
        float sum = 0.0;
        for (int m = 0; m < poolSize && j + m < p; m++) {
            for (int l = 0; l < poolSize && k + l < q; l++) {
                sum += M[i * p * q + (j + m) * q + (k + l)];
            }
        }
        P[i * (p/poolSize) * (q/poolSize) + (j/poolSize) * (q/poolSize) + (k/poolSize)] = sum / (poolSize * poolSize);
    }
}
 



__device__ float softmax(float x, float *arr, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += expf(arr[i]);
    return expf(x) / sum;
}

//For the dense layers, considering we have three of those, with two different
// activation functions, we have to use a conditional  to choose the activation function

__global__ void denseLayer(float *input, float *weights, float *output, int n, int p, Activation activation) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    output[i * p + j] = 0;
    for (int k = 0; k < n; k++)
        output[i * p + j] += input[i * n + k] * weights[k * p + j];

   
    if (activation == TANH) {
        output[i * p + j] = tanhf(output[i * p + j]);
    } else if (activation == SOFTMAX) {
        output[i * p + j] = softmax(output[i * p + j], output + i * p, p);
    }
}

//this function is inspired in the provided MNIST pinrint function
float *generateGrayscaleImage(int imageIndex) {
    int i, j;
    float *img;
    unsigned int magic, nbImg, nbRows, nbCols;
    unsigned char val;
    FILE *fptr;

 
    img = (float *)malloc(HEIGHT * WIDTH * sizeof(float));
 
    if ((fptr = fopen("train-images.idx3-ubyte", "rb")) == NULL) {
        printf("Can't open file");
        exit(1);
    }
 
    fread(&magic, sizeof(int), 1, fptr);
    fread(&nbImg, sizeof(int), 1, fptr);
    fread(&nbRows, sizeof(int), 1, fptr);
    fread(&nbCols, sizeof(int), 1, fptr);
 
    if (imageIndex < 0 || imageIndex >= nbImg) {
        printf("Invalid image index");
        exit(1);
    }

 
    fseek(fptr, 16 + imageIndex * HEIGHT * WIDTH * sizeof(unsigned char), SEEK_SET);
 
    for (i = 0; i < HEIGHT; i++) {
        for (j = 0; j < WIDTH; j++) {
            fread(&val, sizeof(unsigned char), 1, fptr);
            img[i * WIDTH + j] = (float)val;
        }
    }
 
    fclose(fptr);

    return img;
}

void charBckgrndPrint(const char *str, int rgb[3])
{
    printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
    printf("%s\033[0m", str);
}

void printGrayscaleImage(int height, int width, float *img) {
    int row, col;
    const char *str = "  ";
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            float pixel_value = img[row * width + col];
            int rounded_pixel = (int)pixel_value;
            int grayscale_rgb[3] = {rounded_pixel, rounded_pixel, rounded_pixel};
            charBckgrndPrint(str, grayscale_rgb);
        }
        printf("\n");
    }
}


//This function heps us to read the exported .txt weights from the python script
void readArrayFromFile(const char* filename, float* array, int size) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; ++i) {
        if (fscanf(file, "%f", &array[i]) != 1) {
            fprintf(stderr, "Error reading from file: %s\n", filename);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
}