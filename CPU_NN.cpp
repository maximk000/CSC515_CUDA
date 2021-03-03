#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// M - input 1 columns        N - input 1 columns/input 2 rows        K - input 2 columns
void matMultiplyOnHost(float* A, float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            C[i * K + j] = 0.0;
            for (int x = 0; x < N; x++)
            {
                C[i * K + j] += A[i * N + x] * B[x * K + j];
            }
        }
    }
    return;
}

//CPU Matrix Multiplication with Bias added
void matMultiplyOnHostBias(float* A, float* B, float* C, float* bias, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            C[i * K + j] = 0.0;
            for (int x = 0; x < N; x++)
            {
                C[i * K + j] += A[i * N + x] * B[x * K + j];
            }
            C[i * K + j] += bias[i * K + j];
        }
    }
    return;
}

// Relu CPU function
void reluActivationForwardOnHost(float* Z, float* A, int Z_x_dim, int Z_y_dim) {

    for(int index = 0; index < Z_x_dim * Z_y_dim; index++) {
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
}


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
        host_input[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    for (int i = 0; i < layer1_rows * layer1_cols; i++)
    {
        host_layer1w[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    for (int i = 0; i < input_rows * layer1_cols; i++)
    {
        host_layer1b[i] = 1.0;
    }
    for (int i = 0; i < layer2_rows * layer2_cols; i++)
    {
        host_layer2w[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    for (int i = 0; i < input_rows * layer2_cols; i++)
    {
        host_layer2b[i] = 1.0;
    }

    matMultiplyOnHost(host_input, host_layer1w, hostComputedC, input_rows, input_cols, layer1_cols);
    printf("\nMatrix C From Host\n");
    Print_Mat(input_rows, layer1_cols, hostComputedC);//Function Call

    return 0;

}
