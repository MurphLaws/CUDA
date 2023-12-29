#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix_operations.cuh"

int main() {
    
//DECLARATION OF RAW DATA MATRIX
    
    int image_size = 8;
    float *raw_data = (float*)malloc(image_size*image_size*sizeof(float));

    MatrixInit01_2D(raw_data, image_size,image_size);
    //MatrixPrint(raw_data, image_size,image_size);


 //DECLARATION OF C1, with the convolution values

    int number_of_filters = 6;
    int output_size_C1 = 6;

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


    float *dense_weights = (float*)malloc(120*84*sizeof(float));
    float *dense_output = (float*)malloc(84*sizeof(float));

    MatrixInit01_2D(dense_weights, 120, 84);

    //Store all the declared matrices in the GPU memory

    float *d_raw_data, *d_C1_data, *d_S1_data, *d_C1_kernel, *d_dense_weights, *d_dense_output;

    cudaMalloc((void**)&d_raw_data, image_size*image_size*sizeof(float));
    cudaMalloc((void**)&d_C1_data, number_of_filters*output_size_C1*output_size_C1*sizeof(float));
    cudaMalloc((void**)&d_C1_kernel, number_of_filters*kernel_size*kernel_size*sizeof(float));
    cudaMalloc((void**)&d_S1_data, number_of_filters*output_size_S1*output_size_S1*sizeof(float));
    cudaMalloc((void**)&d_dense_weights, 120*84*sizeof(float));
    cudaMalloc((void**)&d_dense_output, 84*sizeof(float));



    cudaMemcpy(d_raw_data, raw_data, image_size*image_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, number_of_filters*output_size_C1*output_size_C1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, number_of_filters*kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, number_of_filters*output_size_S1*output_size_S1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dense_weights, dense_weights, 120*84*sizeof(float), cudaMemcpyHostToDevice);
    //Call the convolution function defined as __global__ void convolution(float *input, int input_size, int kernel_size, int number_of_filters, float *kernel, float *output);

    convolution<<<1,number_of_filters>>>(d_raw_data, image_size, kernel_size, number_of_filters, d_C1_kernel, d_C1_data);

    //Call the activation function defined as __global__ void activation_tanh(float *M, int n);


   

    dim3 grid(6);
    dim3 block(6, 6);


   
    avgPooling<<<grid,block>>>(d_C1_data, d_S1_data, number_of_filters, output_size_C1, output_size_C1, 2);



    //Call the dense layer function defined as __global__ void denseLayer(float *input, float *weights, float *output, int input_size, int output_size);

    denseLayer<<<100,100>>>(d_S1_data, d_dense_weights, d_dense_output, 120, 84);

    //Copy the result of the convolution from the GPU memory to the CPU memory
    cudaMemcpy(dense_output, d_dense_output, 10*sizeof(float), cudaMemcpyDeviceToHost);
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
    printf("-----------------------------------------\n");
    MatrixPrint(dense_output, 84, 1);

    //MatrixPrint3D(S1_data, number_of_filters, output_size_S1, output_size_S1);

    //print the C1_data array using loops. Print is as a 1d array, only using one loop


    
 
}


