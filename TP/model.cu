#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix_operations.cuh"

int main() {
    
//DECLARATION OF RAW DATA MATRIX
    
    int image_size = 6;
    float *raw_data = (float*)malloc(image_size*image_size*sizeof(float));

    MatrixInit01(raw_data, image_size,image_size);
    //MatrixPrint(raw_data, image_size,image_size);


 //DECLARATION OF C1, with the convolution values

    int number_of_filters = 6;
    int output_size_C1 = 28;

    float *C1_data  = (float*)malloc(number_of_filters*output_size_C1*output_size_C1*sizeof(float));
    MatrixInit3D(C1_data , number_of_filters, output_size_C1, output_size_C1);
    MatrixPrint(C1_data , number_of_filters, output_size_C1*output_size_C1);

//DECLARATION OF S1, with the pooling values


    int output_size_S1 = 14;
    float *S1_data = (float*)malloc(number_of_filters*output_size_S1*output_size_S1*sizeof(float));
    MatrixInit3D(S1_data, number_of_filters, output_size_S1, output_size_S1);
    MatrixPrint(S1_data, number_of_filters, output_size_S1*output_size_S1);


// DECLARATION OF C1_kernel 

    int kernel_size = 5;
    float *C1_kernel = (float*)malloc(number_of_filters*kernel_size*kernel_size*sizeof(float));
    MatrixInit3D(C1_kernel, number_of_filters, kernel_size, kernel_size);
    MatrixPrint(C1_kernel, number_of_filters, kernel_size*kernel_size);




}


