#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

#include "swatch.h"

void mv(int n, float *y, float *A, float *x)
// Compute y = Ax where A is an n-by-n matrix, 
// x is an n-dimensional vector, and y is an n-dimensional vector.
// Note that A is represented as a 1-dimensional array stored in the row-major order.
{
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++)
            sum += A[i * n + j] * x[j];
        y[i] = sum;
    }
}

__global__ void mv_kernel_1_1(int n, float *y, float *A, float *x)
// Compute y = Ax with a single thread on a GPU,
// where A is an n-by-n matrix, x is an n-dimensional vector, and y is an n-dimensional vector.
// Note that A is represented as a 1-dimensional array stored in the row-major order.
// This kernel is called like mv_kernel_1_1<<< 1, 1 >>>(n, y, A, x);
{
    // Write by yourself.
}

__global__ void mv_kernel_1_256(int n, float *y, float *A, float *x)
// Compute y = Ax with a single thread block of 256 threads on a GPU,
// where A is an n-by-n matrix, x is an n-dimensional vector, and y is an n-dimensional vector.
// Note that A is represented as a 1-dimensional array stored in the row-major order.
// This kernel is called like mv_kernel_1_256<<< 1, 256 >>>(n, y, A, x);
{
    // Write by yourself.
}

__global__ void mv_kernel(int n, float *y, float *A, float *x)
// Compute y = Ax with multiple thread blocks of 256 threads (the total should be at least n threads) on a GPU,
// where A is an n-by-n matrix, x is an n-dimensional vector, and y is an n-dimensional vector.
// Note that A is represented as a 1-dimensional array stored in the row-major order.
// This kernel is called like mv_kernel<<< (n + 256 - 1) / 256, 256 >>>(n, y, A, x);
{
    // Write by yourself.
}

void checkRelativeError(int n, float* Exact, float* Approx, float epsilon)
/*
  For given n-dimensional vectors Exact and Approx represented as 1D arrays,
  the relative error of each element in Approx to the corresponding element in Exact is checked,
  and "Check: NG" is displayed if any element has a relative error greater than epsilon, 
  and "Check: OK" is displayed otherwise.
  Either column-major order or row-major order is acceptable unless they are mixed.
*/
{
    printf("Check: ");

    for (int i = 0; i < n; i++)
        if (fabsf(Approx[i] - Exact[i]) / fabsf(Exact[i]) > epsilon) {
            printf("NG\n");
            return;
        }

    printf("OK\n");
}

const int N = 1024;
float A[N * N], x[N], y_cpu[N], y_gpu[N];

int main(int argc, char* argv[])
{
    for (int i = 0; i < N * N; i++)
        A[i] = (float) rand() / RAND_MAX;
    for (int i = 0; i < N; i++)
        x[i] = (float) rand() / RAND_MAX;

    mv(N, y_cpu, A, x);

    printf("CPU:\n");
    for (int i = 0; i < 4; i++) printf("y[%d] = %f\n", i, y_cpu[i]);
    for (int i = N - 4; i < N; i++) printf("y[%d] = %f\n", i, y_cpu[i]);

    float *d_A, *d_x, *d_y;

    // Here, Do cudaMalloc() N * N floats to d_A.
    // Here, Do cudaMalloc() N floats to d_x.
    // Here, Do cudaMalloc() N floats to d_y.

    // Here, copy A[] to device memory pointed by d_A with cudaMemcpy()
    // Here, copy x[] to device memory pointed by d_x with cudaMemcpy()

    // Here, execute kernel function mv_kernel_1_1(), mv_kernel_1_256(), or mv_kernel()

    // Here, copy device memory pointed by d_y to y_gpu[] with cudaMemcpy()

    // Here, Do cudaFree() for d_A.
    // Here, Do cudaFree() for d_x.
    // Here, Do cudaFree() for d_y.

    printf("GPU:\n");
    for (int i = 0; i < 4; i++) printf("y[%d] = %f\n", i, y_gpu[i]); 
    for (int i = N - 4; i < N; i++) printf("y[%d] = %f\n", i, y_gpu[i]);

    checkRelativeError(N, y_cpu, y_gpu, 1e-6);

    cudaDeviceReset();

    return 0;
}
