#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <immintrin.h>  /* AVX命令の組み込み関数用ヘッダ */

#include "swatch.h"     /* 時間測定用 */

void colMajorMatrixMulAVX(int n, float* A, float* B, float* C)
/*
  列優先の1次元配列で表現されたn次正方行列A, B, Cに対して，
  C += ABを計算する．
  nは8の倍数と仮定する．
*/
{
    for (int i = 0; i < n; i += 8)   /* ストリップマイニングしていることに注意 */
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
  列優先の1次元配列で表現されたn次正方行列A, B, Cに対して，
  C += ABを計算する．
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
  1次元配列で表現されたn次正方行列Aの各要素を
  0以上1以下のランダムな値に初期化する．
  Aは行優先と列優先のどちらでもよい．
*/
{
    for (int i = 0; i < n * n; i++)
        A[i] = (float) rand() / RAND_MAX;
}

void clearMatrix(int n, float* A)
/*
  1次元配列で表現されたn次正方行列Aの全要素を0にする．
  Aは行優先と列優先のどちらでもよい．
*/
{
    for (int i = 0; i < n * n; i++)
        A[i] = 0.0f;
}

void checkRelativeError(int n, float* Exact, float* Approx, float epsilon)
/*
  1次元配列で表現されたn次正方行列Exact, Approxに対して，
  Approxの各要素のExactの対応する要素に対する相対誤差をチェックし，
  相対誤差が epsilon を超える要素がひとつでもあれば "Check: NG" と表示し，
  全要素の相対誤差が epsilon 以下であれば "Check: OK" と表示する．

  Exact, Approxは両方とも行優先と両方とも列優先のどちらでもよい．
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
  列優先の1次元配列で表現されたn次正方行列Aの内容を表示する．
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
  行優先の1次元配列で表現されたn次正方行列Aの内容を表示する．
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
  行優先の1次元配列で表現されたn次正方行列A, B, Cに対して，
  C += ABを計算する．
  nは8の倍数と仮定する．
*/
{
    for (int j = 0; j < n; j += 8)   /* ストリップマイニングしていることに注意, 列の */
        for (int i = 0; i < n; i++) {
            __m256 c0 = _mm256_setzero_ps();
            for (int k = 0; k < n; k++)
                c0 = _mm256_add_ps(
                    c0, /* c0[h] += A[i][k] * B[k][j] (0 <= h < 8) */
                    _mm256_mul_ps(
                        //行優先ではAの行を固定（ブロードキャスト）して，Bを回す．
                        _mm256_load_ps(B + k * n + j), //行が1増えるとn増やさないといけないのでk * n，列が1増えると1増やさないといけないので+j
                        _mm256_broadcast_ss(A + i * n + k)//行が1増えるとn増やさないといけないのでi * n，列が1増えると1増やさないといけないので+k
                    )
                );
            _mm256_store_ps(C + i * n + j, c0);   /* C[i][j + h] = c0[h] (0 <= h < 8), 注目箇所の一つ上の行までにi*n個，注目箇所と同じ行にj個登場するのでC + i * n + jとなる */
        }
}

void rowMajorMatrixMul(int n, float* A, float* B, float* C)
/*
  行優先の1次元配列で表現されたn次正方行列A, B, Cに対して，
  C += ABを計算する．
*/
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            float cij = 0;
            for (int k = 0; k < n; k++)
                cij += A[i * n + k] * B[k * n + j];    /* cij += A[i][k] * B[k][j] */
            C[i * n + j] = cij;   /* C[i][j] = cij */
        }
}
const int N = 1024;  /* 8の倍数でなければならない */

/* 配列A, B, C2は，32バイト境界に配置しなければならない */
__attribute__((aligned(32))) float A[N * N], B[N * N], C1[N * N], C2[N * N];  

int main(void)
{
    srand(2022);  // 実験の再現性を確保するため，種を定数にする．
    randomizeMatrix(N, A);
    randomizeMatrix(N, B);
    clearMatrix(N, C1);
    clearMatrix(N, C2);

    printf("現在使用中のスレッド数は「%d」です。\n", omp_get_num_threads() );

    StopWatch sw;

    sw.Reset();
    sw.Start();
    rowMajorMatrixMul(N, A, B, C1);
    sw.Stop();
    printf("Standard C: %lf sec\n", sw.GetTime());

    sw.Reset();
    sw.Start();
    rowMajorMatrixMulAVX(N, A, B, C2);
    sw.Stop();
    printf("AVX intrinsics: %lf sec\n", sw.GetTime());

    checkRelativeError(N, C1, C2, 1e-6);

    if (N <= 8) {
        printf("---------------\n");
        printf("A:\n");
        printRowMajorMatrix(N, A);
        printf("---------------\n");
        printf("B:\n");
        printRowMajorMatrix(N, B);
        printf("---------------\n");
        printf("C1:\n");
        printRowMajorMatrix(N, C1);
        printf("---------------\n");
        printf("C2:\n");
        printRowMajorMatrix(N, C2);
    }

    return 0;
}
