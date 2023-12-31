#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix_operations.cuh"

#define WIDTH 28
#define HEIGHT 28



int main(int argc, char* argv[]) {
        if (argc != 2) {
            fprintf(stderr, "Usage: %s MNISTidx\n", argv[0]);
            return 1;
        }

    int MNISTidx = atoi(argv[1]);


    float *d_raw_data, *d_C1_data, *d_S2_data, *d_C1_kernel, *d_C3_data,*d_C3_kernel, *d_S4_data, *d_C5_data, *d_C5_weights, *d_C6_data, *d_C6_weights, *d_C7_data, *d_C7_weights;


    //INPUT IMAGE
    int image_size = 28;
    float *raw_data = (float*)malloc(image_size*image_size*sizeof(float));

    raw_data = generateGrayscaleImage(MNISTidx);

    //C1 DECLARATION

    int C1_filters = 6;
    int output_size_C1 = 24;

    float *C1_data  = (float*)malloc(C1_filters*output_size_C1*output_size_C1*sizeof(float));
    MatrixInit0_3D(C1_data , C1_filters, output_size_C1, output_size_C1);

    int kernel_size = 5;
    float *C1_kernel = (float*)malloc(C1_filters*kernel_size*kernel_size*sizeof(float));

    readArrayFromFile("C1.txt", C1_kernel, C1_filters*kernel_size*kernel_size);
   

    //S2 DECLARATION

    int output_size_S2 = output_size_C1/2;
    float *S2_data = (float*)malloc(C1_filters*output_size_S2*output_size_S2*sizeof(float));
    MatrixInit0_3D(S2_data, C1_filters, output_size_S2, output_size_S2);

    //C3 DECLARATION

    int C3_filters = 16*C1_filters;
    int output_size_C3 = 8;
    float *C3_data = (float*)malloc(C3_filters*output_size_C3*output_size_C3*sizeof(float));
    MatrixInit0_3D(C3_data, C3_filters, output_size_C3, output_size_C3); 

    float *C3_kernel = (float*)malloc(C3_filters*kernel_size*kernel_size*sizeof(float));
    
    readArrayFromFile("C3.txt", C3_kernel, C3_filters*kernel_size*kernel_size);


    //S4 DECLARATION

    int output_size_S4 = output_size_C3/2;
    float *S4_data = (float*)malloc(C3_filters*output_size_S4*output_size_S4*sizeof(float));
    MatrixInit0_3D(S4_data, C3_filters, output_size_S4, output_size_S4);

    // C5 

    int C5_input = 256;
    int C5_output = 120;

    float *C5_data = (float*)malloc(C5_output*sizeof(float));
    MatrixInit0(C5_data, C5_output);

    float *C5_weights = (float*)malloc(C5_input*C5_output*sizeof(float));
    
    readArrayFromFile("C5.txt", C5_weights, C5_input*C5_output);

    // C6

    int C6_input = 120;
    int C6_output = 84;

    float *C6_data = (float*)malloc(C6_output*sizeof(float));
    MatrixInit0(C6_data, C6_output);

    float *C6_weights = (float*)malloc(C6_input*C6_output*sizeof(float));

    readArrayFromFile("C6.txt", C6_weights, C6_input*C6_output);
    
    

    // C7

    int C7_input = 84;
    int C7_output = 10;

    float *C7_data = (float*)malloc(C7_output*sizeof(float));
    MatrixInit0(C7_data, C7_output);

    float *C7_weights = (float*)malloc(C7_input*C7_output*sizeof(float));
    
    readArrayFromFile("C7.txt", C7_weights, C7_input*C7_output);






    

    cudaMalloc((void**)&d_raw_data, image_size*image_size*sizeof(float));
    cudaMalloc((void**)&d_C1_data, C1_filters*output_size_C1*output_size_C1*sizeof(float));
    cudaMalloc((void**)&d_C1_kernel, C1_filters*kernel_size*kernel_size*sizeof(float));
    cudaMalloc((void**)&d_S2_data, C1_filters*output_size_S2*output_size_S2*sizeof(float));
    cudaMalloc((void**)&d_C3_kernel, C3_filters*C1_filters*kernel_size*kernel_size*sizeof(float));
    cudaMalloc((void**)&d_C3_data, C3_filters*output_size_C3*output_size_C3*sizeof(float));
    cudaMalloc((void**)&d_S4_data, C3_filters*output_size_S4*output_size_S4*sizeof(float));
    cudaMalloc((void**)&d_C5_data, C5_output*sizeof(float));
    cudaMalloc((void**)&d_C5_weights, C5_input*C5_output*sizeof(float));
    cudaMalloc((void**)&d_C6_data, C6_output*sizeof(float));
    cudaMalloc((void**)&d_C6_weights, C6_input*C6_output*sizeof(float));
    cudaMalloc((void**)&d_C7_data, C7_output*sizeof(float));
    cudaMalloc((void**)&d_C7_weights, C7_input*C7_output*sizeof(float));

    cudaMemcpy(d_raw_data, raw_data, image_size*image_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, C1_filters*output_size_C1*output_size_C1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, C1_filters*kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S2_data, S2_data, C1_filters*output_size_S2*output_size_S2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3_data, C3_data, C3_filters*output_size_C3*output_size_C3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3_kernel, C3_kernel, C3_filters*C1_filters*kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S4_data, S4_data, C3_filters*output_size_S4*output_size_S4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C5_data, C5_data, C5_output*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C5_weights, C5_weights, C5_input*C5_output*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C6_data, C6_data, C6_output*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C6_weights, C6_weights, C6_input*C6_output*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C7_data, C7_data, C7_output*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C7_weights, C7_weights, C7_input*C7_output*sizeof(float), cudaMemcpyHostToDevice);


    //C1
    convolution<<<1,C1_filters>>>(d_raw_data, image_size, kernel_size, C1_filters, d_C1_kernel, d_C1_data);
    //C2
    dim3 grid(C1_filters);
    dim3 block(output_size_S2, output_size_S2);
    avgPooling<<<grid,block>>>(d_C1_data, d_S2_data, C1_filters, output_size_C1, output_size_C1, 2);
    //C3
    convolution<<<1,C3_filters*2>>>(d_S2_data, output_size_S2, kernel_size, C3_filters, d_C3_kernel, d_C3_data);

    //S4
    dim3 grid2(C3_filters);
    dim3 block2(output_size_S4, output_size_S4);
    avgPooling<<<grid2,block2>>>(d_C3_data, d_S4_data, C3_filters, output_size_C3, output_size_C3, 2);

    //C5
  
    denseLayer<<<1,C5_output>>>(d_S4_data, d_C5_weights, d_C5_data, C5_input, C5_output,TANH );

    //C6
    denseLayer<<<1,C6_output>>>(d_C5_data, d_C6_weights, d_C6_data, C6_input, C6_output, TANH);

    //C7
    denseLayer<<<1,C7_output>>>(d_C6_data, d_C7_weights, d_C7_data, C7_input, C7_output, SOFTMAX);



    cudaMemcpy(C1_data, d_C1_data, C1_filters*output_size_C1*output_size_C1*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(S2_data, d_S2_data, C1_filters*output_size_S2*output_size_S2*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C3_data, d_C3_data, C3_filters*output_size_C3*output_size_C3*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(S4_data, d_S4_data, C3_filters*output_size_S4*output_size_S4*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C5_data, d_C5_data, C5_output*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C6_data, d_C6_data, C6_output*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C7_data, d_C7_data, C7_output*sizeof(float), cudaMemcpyDeviceToHost);



    //void printGrayscaleImage(int height, int width, float *img) 
    printGrayscaleImage(28, 28, raw_data);
    MatrixPrint(C7_data, 10, 1);
 
   


    //MEMORY FREEING

    free(raw_data);
    free(C1_data);
    free(C1_kernel);
    free(S2_data);
    free(C3_data);
    free(C3_kernel);
    free(S4_data);
    free(C5_data);
    free(C5_weights);
    free(C6_data);
    free(C6_weights);
    free(C7_data);
    free(C7_weights);

    cudaFree(d_raw_data);
    cudaFree(d_C1_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_S2_data);
    cudaFree(d_C3_data);
    cudaFree(d_C3_kernel);
    cudaFree(d_S4_data);
    cudaFree(d_C5_data);
    cudaFree(d_C5_weights);
    cudaFree(d_C6_data);
    cudaFree(d_C6_weights);
    cudaFree(d_C7_data);
    cudaFree(d_C7_weights);

    
    return EXIT_SUCCESS;



}