#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix_operations.cuh"

int main() {
    
//DECLARATION OF RAW DATA MATRIX
    
    int image_size = 6;
    float *raw_data = (float*)malloc(image_size*image_size*sizeof(float));

    MatrixInit01_2D(raw_data, image_size,image_size);
    //MatrixPrint(raw_data, image_size,image_size);


 //DECLARATION OF C1, with the convolution values

    int number_of_filters = 6;
    int output_size_C1 = 4;

    float *C1_data  = (float*)malloc(number_of_filters*output_size_C1*output_size_C1*sizeof(float));
    MatrixInit0_3D(C1_data , number_of_filters, output_size_C1, output_size_C1);

//DECLARATION OF S1, with the pooling values


    int output_size_S1 = output_size_C1/2;
    float *S1_data = (float*)malloc(number_of_filters*output_size_S1*output_size_S1*sizeof(float));
    MatrixInit0_3D(S1_data, number_of_filters, output_size_S1, output_size_S1);


// DECLARATION OF C1_kernel 

    int kernel_size = 3;
    float *C1_kernel = (float*)malloc(number_of_filters*kernel_size*kernel_size*sizeof(float));
    MatrixInit01_3D(C1_kernel, number_of_filters, kernel_size, kernel_size);



    //Store all the declared matrices in the GPU memory

    float *d_raw_data, *d_C1_data, *d_S1_data, *d_C1_kernel;

    cudaMalloc((void**)&d_raw_data, image_size*image_size*sizeof(float));
    cudaMalloc((void**)&d_C1_data, number_of_filters*output_size_C1*output_size_C1*sizeof(float));
    cudaMalloc((void**)&d_C1_kernel, number_of_filters*kernel_size*kernel_size*sizeof(float));
    cudaMalloc((void**)&d_S1_data, number_of_filters*output_size_S1*output_size_S1*sizeof(float));

    cudaMemcpy(d_raw_data, raw_data, image_size*image_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, number_of_filters*output_size_C1*output_size_C1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, number_of_filters*kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice);
    

    //Call the convolution function defined as __global__ void convolution(float *input, int input_size, int kernel_size, int number_of_filters, float *kernel, float *output);


    convolution<<<1, number_of_filters>>>(d_raw_data, image_size, kernel_size, number_of_filters, d_C1_kernel, d_C1_data);

    //void __global__ void subsampling(float *input, int input_size, int kernel_size, float *output);

    // Define a 2x2 subsampling kernel
    float subsampling_kernel[4] = {0.25, 0.25, 0.25, 0.25};

// Allocate memory on the device for the subsampling kernel
    float *d_subsampling_kernel;
    cudaMalloc(&d_subsampling_kernel, 4 * sizeof(float));

    // Copy the subsampling kernel to the device
    cudaMemcpy(d_subsampling_kernel, subsampling_kernel, 4 * sizeof(float), cudaMemcpyHostToDevice);


    convolution<<<1, number_of_filters>>>(d_C1_data, output_size_C1, 2, 1, d_subsampling_kernel, d_S1_data);


    
    //store the result of the convolution in the host memory

    cudaMemcpy(C1_data, d_C1_data, number_of_filters*output_size_C1*output_size_C1*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, number_of_filters*output_size_S1*output_size_S1*sizeof(float), cudaMemcpyDeviceToHost);

    //print the result of the convolution using the MatrixPrint3D function
    MatrixPrint(raw_data, image_size,image_size);
    printf("-----------------------------------------\n");
    MatrixPrint3D(C1_kernel, number_of_filters, kernel_size, kernel_size);
    printf("-----------------------------------------\n");
    MatrixPrint3D(C1_data, number_of_filters, output_size_C1, output_size_C1);
    printf("-----------------------------------------\n");
    MatrixPrint3D(S1_data, number_of_filters, output_size_S1, output_size_S1);


    
 
}


