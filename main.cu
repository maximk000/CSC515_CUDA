#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <driver_types.h>
#include "device_launch_parameters.h"

#define Tile_size 2

//Function To handle any errors occurred in the function calls


// Compute C = A * B
//*************************************************************
//Kernel for shared memory/ Tiled execution
__global__ void matrixMultiplyShared(float* A, float* B, float* C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns)
{
    __shared__ float sA[Tile_size][Tile_size];   // Tile size to store elements in shared memory
    __shared__ float sB[Tile_size][Tile_size];

    int Row = blockDim.y * blockIdx.y + threadIdx.y; //To generate ids of threads.
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int k = 0; k < (((numAColumns - 1) / Tile_size) + 1); k++)
    {
        if ((Row < numARows) && (threadIdx.x + (k * Tile_size)) < numAColumns)//Copy Data to Tile from Matrix (Global Memory to Shared Memory)
        {
            sA[threadIdx.y][threadIdx.x] = A[(Row * numAColumns) + threadIdx.x + (k * Tile_size)];
        }
        else
        {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (Col < numBColumns && (threadIdx.y + k * Tile_size) < numBRows)//Copy Data to Tile from Matrix (Global Memory to Shared Memory)
        {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k * Tile_size) * numBColumns + Col];
        }
        else
        {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < Tile_size; ++j)//Multiplying Elements present in tile
        {
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (Row < numCRows && Col < numCColumns)//Saving Final result into Matrix C
    {
        C[Row * numCColumns + Col] = Cvalue;
    }
}

__global__ void reluActivationForward(float* Z, float* A,
    int Z_x_dim, int Z_y_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < Z_x_dim * Z_y_dim) {
        A[index] = fmaxf(Z[index], 0);
    }
}

//*************************************************************
void Print_Mat(int Row, int Col, float* Mat)//Function To print the Matrix
{
    for (int i = 0; i < Row * Col; i++)
    {
        printf("%f  ", *(Mat + i));

        if ((i % Col) == 0)
        {
            printf("\n");
        }
    }
}//Function close
//*************************************************************
//Normal CPU Matrix Multiplication
void matMultiplyOnHost(float* A, float* B, float* C, int numARows,
    int numAColumns, int numBRows, int numBColumns,
    int numCRows, int numCColumns)
{
    for (int i = 0; i < numARows; i++)
    {
        for (int j = 0; j < numBColumns; j++)
        {
            C[i * numCColumns + j] = 0.0;
            for (int k = 0; k < numBRows; k++)
            {
                C[i * numCColumns + j] += A[i * numAColumns + k] * B[k * numBColumns + j];
            }
        }
    }
    return;
}
//*************************************************************
int input_rows = 1;
int input_cols = 784;
int layer1_rows = 784;
int layer1_cols = 128;
int layer2_rows = 128;
int layer2_cols = 10;
int output_rows = input_rows;
int output_cols = 10;

int main(int argc, char** argv) {
    float* host_input;
    float* host_output;

    float* host_layer1w;
    float* host_layer1b;
    float* host_layer1out;

    float* host_layer2w;
    float* host_layer2b;

    float* hostComputedC;

    float* device_input;
    float* device_output;

    float* device_layer1w;
    float* device_layer1b;
    float* device_layer1out_w;
    float* device_layer1out_b;

    float* device_layer2w;
    float* device_layer2b;

    // Please adjust rows and columns according to you need.

    //printf("\nPlease Enter Rows and Columns of A:");
    //scanf("%d %d", &numARows, &numAColumns);

    //printf("\nPlease Enter Rows and Columns of B:");
    //scanf("%d %d", &numBRows, &numBColumns);

    host_input = (float*)malloc(sizeof(float) * input_rows * input_cols);
    host_output = (float*)malloc(sizeof(float) * output_rows * output_cols);
    host_layer1w = (float*)malloc(sizeof(float) * layer1_rows * layer1_cols);
    host_layer1b = (float*)malloc(sizeof(float) * input_rows * layer1_cols);
    host_layer1out = (float*)malloc(sizeof(float) * input_rows * layer1_cols);
    host_layer2w = (float*)malloc(sizeof(float) * layer2_rows * layer2_cols);
    host_layer2b = (float*)malloc(sizeof(float) * input_rows * layer2_cols);
    hostComputedC = (float*)malloc(sizeof(float) * input_rows * layer1_cols);

    for (int i = 0; i < input_rows * input_cols; i++)//Matrix Initialization
    {
        host_input[i] = 1.0;
    }
    for (int i = 0; i < layer1_rows * layer1_cols; i++)
    {
        host_layer1w[i] = 1.0;
    }
    for (int i = 0; i < input_rows * layer1_cols; i++)
    {
        host_layer1b[i] = 1.0;
    }
    for (int i = 0; i < layer2_rows * layer2_cols; i++)
    {
        host_layer2w[i] = 1.0;
    }
    for (int i = 0; i < input_rows * layer2_cols; i++)
    {
        host_layer2b[i] = 1.0;
    }

    printf("\nMatrix A Values:\n");
    //Print_Mat(numARows, numAColumns, hostA);//Function Call

    printf("\n\nMatrix B Values:\n");
    //Print_Mat(numBRows, numBColumns, hostB);//Function Call

    // Allocating GPU memory
    cudaMalloc((void**)&device_input, sizeof(float) * input_rows * input_cols);

    //Output vector
    cudaMalloc((void**)&device_output, sizeof(float) * output_rows * output_cols);

    //Layer 1
    cudaMalloc((void**)&device_layer1w, sizeof(float) * layer1_rows * layer1_cols);
    cudaMalloc((void**)&device_layer1b, sizeof(float) * input_rows * layer1_cols);
    cudaMalloc((void**)&device_layer1out_w, sizeof(float) * input_rows * layer1_cols);
    cudaMalloc((void**)&device_layer1out_b, sizeof(float) * input_rows * layer1_cols);

    //Layer 2
    cudaMalloc((void**)&device_layer2w, sizeof(float) * layer2_rows * layer2_cols);
    cudaMalloc((void**)&device_layer2b, sizeof(float) * input_rows * layer2_cols);

    // Copy memory to the GPU
    cudaMemcpy(device_input, host_input, sizeof(float) * input_rows * input_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_output, host_output, sizeof(float) * output_rows * output_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_layer1w, host_layer1w, sizeof(float) * layer1_rows * layer1_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_layer1b, host_layer1b, sizeof(float) * input_rows * layer1_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_layer2w, host_layer2w, sizeof(float) * layer2_rows * layer2_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(device_layer2b, host_layer2b, sizeof(float) * input_rows * layer2_cols, cudaMemcpyHostToDevice);

    // Initialize the grid and block dimensions

    dim3 dimGrid((layer1_cols / Tile_size) + 1, (input_rows / Tile_size) + 1, 1);//Number of Blocks required
    dim3 dimBlock(Tile_size, Tile_size, 1);//Number of threads in each block

    cudaEvent_t start, stop;
    float time_w, time_b;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    //@@ Launch the GPU Kernel here
    cudaEventRecord(start, 0);      // start time measurement
    matrixMultiplyShared << <dimGrid, dimBlock >> > (device_input, device_layer1w, device_layer1out_w, input_rows, input_cols, layer1_rows, layer1_cols, input_rows, layer1_cols);


    cudaDeviceSynchronize();//To synchronize the device
    cudaEventRecord(stop, 0);       // stop time measurement
    cudaEventSynchronize(stop);     // sync results
    cudaEventElapsedTime(&time_w, start, stop);
    printf("Elapsed time : %f ms\n", time_w);

    cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

  

    // Copy the results in GPU memory back to the CPU
    cudaMemcpy(host_layer1out, device_layer1out_w, sizeof(float) * input_rows * layer1_cols, cudaMemcpyDeviceToHost);

    printf("\nMatrix C From Device\n");
    //Print_Mat(numCRows, numCColumns, hostC);//Function Call

    matMultiplyOnHost(host_input, host_layer1w, hostComputedC, input_rows, input_cols, layer1_rows, layer1_cols, input_rows, layer1_cols);

    printf("\nMatrix C From Host\n");
    //Print_Mat(numCRows, numCColumns, hostComputedC);//Function Call

    for (int i = 0; i < input_rows * layer1_cols; i++)//Compare both the result matrices 1. MatrixMultiplyonHost 2. MatrixMultiplyonDevice
    {
        if (hostComputedC[i] != host_layer1out[i])
        {
            printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i / layer1_cols, i % layer1_cols, hostComputedC[i], host_layer1out[i]);
            break;
        }
    }

    printf("\n Number of Blocks Created:%d \n", ((layer1_cols / Tile_size) + 1) * ((layer1_cols / Tile_size) + 1));
    printf("\n Number of Threads Per Block: %d \n", (Tile_size * Tile_size));
    //return 0;
    // Free the GPU memory
    free(host_input);
    free(host_output);
    free(host_layer1w);
    free(host_layer1b);
    free(host_layer1out);
    free(host_layer2w);
    free(host_layer2b);
    free(hostComputedC);

    cudaFree(device_input);
    cudaFree(device_output);

    cudaFree(device_layer1w);
    cudaFree(device_layer1b);
    cudaFree(device_layer1out_w);
    cudaFree(device_layer1out_b);

    cudaFree(device_layer2w);
    cudaFree(device_layer2b);
    int i = 1;
    //while (i > 0) {
        //printf("m");
    //}
    return 0;
}