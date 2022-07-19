#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

#include "swatch.h"

void mv(int n, float *y, float *A, float *x)
// 行列Aとベクトルxの積をベクトルyに求める。 
// 行列Aが行優先の1次元配列で表現されていることに注意。
{
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++)
            sum += A[i * n + j] * x[j];
        y[i] = sum;
    }
}

__global__ 
void mv_kernel_1_1(int n, float *y, float *A, float *x)
// GPU上で1スレッドを用いて
// 行列Aとベクトルxの積をベクトルyに求める。 
// 行列Aが行優先の1次元配列で表現されていることに注意。
// このカーネルは次のように呼び出される：mv_kernel_1_256<<< 1, 1 >>>(n, y, A, x);
{
    // 自分で書く。
    // int index_col = threadIdx.x; //index of the current thread within its block
    // int stride = blockDim.x; //the number of threads in the block
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i+=stride) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++)
            sum += A[i * n + j] * x[j];
        y[i] = sum;
    }
}

__global__ void mv_kernel_1_256(int n, float *y, float *A, float *x)
// GPU上で256スレッドのスレッドブロック1つ用いて
// 行列Aとベクトルxの積をベクトルyに求める。 
// 行列Aが行優先の1次元配列で表現されていることに注意。
// このカーネルは次のように呼び出される：mv_kernel_1_256<<< 1, 256 >>>(n, y, A, x);
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i+=stride) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++)
            sum += A[i * n + j] * x[j];
        y[i] = sum;
    }
}

__global__ void mv_kernel(int n, float *y, float *A, float *x)
// GPU上で256スレッドのスレッドブロックを複数用いて（少なくともnスレッドを用いて）
// 行列Aとベクトルxの積をベクトルyに求める。 
// 行列Aが行優先の1次元配列で表現されていることに注意。
// このカーネルは次のように呼び出される：mv_kernel_1_256<<< (n + 256 - 1), 256 >>>(n, y, A, x);
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i+=stride) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++)
            sum += A[i * n + j] * x[j];
        y[i] = sum;
    }
}

void checkRelativeError(int n, float* Exact, float* Approx, float epsilon)
/*
  1次元配列で表現されたn次元ベクトルExact, Approxに対して，
  Approxの各要素のExactの対応する要素に対する相対誤差をチェックし，
  相対誤差が epsilon を超える要素がひとつでもあれば "Check: NG" と表示し，
  全要素の相対誤差が epsilon 以下であれば "Check: OK" と表示する．
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
    StopWatch sw;

    sw.Reset();
    sw.Start();
    mv(N, y_cpu, A, x);
    sw.Stop();
    printf("CPU: %lf sec\n", sw.GetTime());
    

    printf("CPU:\n");
    for (int i = 0; i < 4; i++) printf("y[%d] = %f\n", i, y_cpu[i]);
    for (int i = N - 4; i < N; i++) printf("y[%d] = %f\n", i, y_cpu[i]);

    float *d_A, *d_x, *d_y;

    // N * N 個の float を cudaMalloc() で d_A に割り当てる。
    // N 個の float を cudaMalloc() で d_x に割り当てる。
    // N 個の float を cudaMalloc() で d_y に割り当てる。
    cudaMalloc((void **)&d_A, N*N*sizeof(float));
    cudaMalloc((void **)&d_x, N*sizeof(float));
    cudaMalloc((void **)&d_y, N*sizeof(float));

    // cudaMemcpy()を用いて A[] を d_A が指すデバイスメモリに転送する。
    // cudaMemcpy()を用いて x[] を d_x が指すデバイスメモリに転送する。
    // cudaMemcpy(dst, src, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);

    // カーネル関数 mv_kernel_1_1() または mv_kernel_1_256() または mv_kernel() を実行する。
    
    // sw.Reset();
    // sw.Start();
    // mv_kernel_1_1<<< 1, 1 >>>(N, d_y, d_A, d_x);
    // sw.Stop();
    // printf("GPU1-1: %lf sec\n", sw.GetTime());

    sw.Reset();
    sw.Start();
    mv_kernel<<< (N + 256 - 1), 256 >>>(N, d_y, d_A, d_x);
    sw.Stop();
    printf("GPU: %lf sec\n", sw.GetTime());

    // cudaMemcpy()を用いて d_y[] を y_gpu[] に転送する。
    cudaMemcpy(y_gpu, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    
    // cudaFree() を用いて d_A が指すメモリ領域を解放する。
    // cudaFree() を用いて d_x が指すメモリ領域を解放する。
    // cudaFree() を用いて d_y が指すメモリ領域を解放する。
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    

    printf("GPU:\n");
    for (int i = 0; i < 4; i++) printf("y[%d] = %f\n", i, y_gpu[i]); 
    for (int i = N - 4; i < N; i++) printf("y[%d] = %f\n", i, y_gpu[i]);
    checkRelativeError(N, y_cpu, y_gpu, 1e-6);

    cudaDeviceReset();

    return 0;
}
