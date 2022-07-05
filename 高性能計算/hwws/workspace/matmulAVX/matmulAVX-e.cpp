#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <immintrin.h>  /* required for AVX intrinsics */

#include "swatch.h"     /* to measure computing time */

void colMajorMatrixMulAVX(int n, float* A, float* B, float* C)
/*
  computes C = AB where A, B, and C are n-by-n matrices stored in column-major order.
  For simplicity, we assume n is a multiple of eight.
*/
{
    for (int i = 0; i < n; i += 8)   /* strip mining */
        for (int j = 0; j < n; j++) {
            __m256 c0 = _mm256_setzero_ps();
            for (int k = 0; k < n; k++)
                c0 = _mm256_add_ps(
                    c0, /* c0[h] += A[i + h][k] * B[k][j] (0 <= h < 8) */
                    _mm256_mul_ps(
                        _mm256_load_ps(A + i + k * n),
                        _mm256_broadcast_ss(B + k + j * n)
                    )
                );
            _mm256_store_ps(C + i + j * n, c0);   /* C[i + h][j] = c0[h] (0 <= h < 8) */
        }
}

void colMajorMatrixMul(int n, float* A, float* B, float* C)
/*
  computes C = AB where A, B, and C are n-by-n matrices stored in column-major order.
*/
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float cij = 0;
            for (int k = 0; k < n; k++)
                cij += A[i + k * n] * B[k + j * n];    /* cij += A[i][k] * B[k][j] */
            C[i + j * n] = cij;   /* C[i][j] = cij */
        }
}

void randomizeMatrix(int n, float* A)
/*
  initializes each element of n-by-n matrix A with a random value in [0, 1]. 
  Either column-major order or row-major order is acceptable.
*/
{
    for (int i = 0; i < n * n; i++)
        A[i] = (float) rand() / RAND_MAX;
}

void clearMatrix(int n, float* A)
/*
  initializes each element of n-by-n matrix A with zero. 
  Either column-major order or row-major order is acceptable.
*/
{
    for (int i = 0; i < n * n; i++)
        A[i] = 0.0f;
}

void checkRelativeError(int n, float* Exact, float* Approx, float epsilon)
/*
  For given n-by-n square matrices Exact and Approx represented as 1D arrays,
  the relative error of each element in Approx to the corresponding element in Exact is checked,
  and "Check: NG" is displayed if any element has a relative error greater than epsilon, 
  and "Check: OK" is displayed otherwise.
  Either column-major order or row-major order is acceptable unless they are mixed.
*/
{
    printf("Check: ");

    for (int i = 0; i < n * n; i++)
        if (fabsf(Approx[i] - Exact[i]) / fabsf(Exact[i]) > epsilon) {
            printf("NG\n");
            return;
        }

    printf("OK\n");
}

void printColMajorMatrix(int n, float* A)
/*
  prints n-by-n matrix A stored in column-major order.
*/
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%f, ", A[i + j * n]);
        printf("\n");
    }
}

void printRowMajorMatrix(int n, float* A)
/*
  prints n-by-n matrix A stored in row-major order.
*/
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%f, ", A[i * n + j]);
        printf("\n");
    }
}

void rowMajorMatrixMulAVX(int n, float* A, float* B, float* C)
/*
  computes C = AB where A, B, and C are n-by-n matrices stored in row-major order.
  For simplicity, we assume n is a multiple of eight.
*/
{
    // Write by yourself.
}

void rowMajorMatrixMul(int n, float* A, float* B, float* C)
/*
  computes C = AB where A, B, and C are n-by-n matrices stored in column-major order.
*/
{
    // Write by yourself.
}

const int N = 1024;  // For simplicity, N must be a multiple of eight

// Arrays A, B, and C2 must be aligned on the 32-byte boundary.
__attribute__((aligned(32))) float A[N * N], B[N * N], C1[N * N], C2[N * N]; 

int main(void)
{
    srand(2022);  // We fix the seed to ensure the reproducibility of experimental results.
    randomizeMatrix(N, A);
    randomizeMatrix(N, B);
    clearMatrix(N, C1);
    clearMatrix(N, C2);

    StopWatch sw;

    sw.Reset();
    sw.Start();
    colMajorMatrixMul(N, A, B, C1);
    sw.Stop();
    printf("Standard C: %lf sec\n", sw.GetTime());

    sw.Reset();
    sw.Start();
    colMajorMatrixMulAVX(N, A, B, C2);
    sw.Stop();
    printf("AVX intrinsics: %lf sec\n", sw.GetTime());

    checkRelativeError(N, C1, C2, 1e-6);

    if (N <= 8) {
        printf("---------------\n");
        printf("A:\n");
        printColMajorMatrix(N, A);
        printf("---------------\n");
        printf("B:\n");
        printColMajorMatrix(N, B);
        printf("---------------\n");
        printf("C1:\n");
        printColMajorMatrix(N, C1);
        printf("---------------\n");
        printf("C2:\n");
        printColMajorMatrix(N, C2);
    }

    return 0;
}
