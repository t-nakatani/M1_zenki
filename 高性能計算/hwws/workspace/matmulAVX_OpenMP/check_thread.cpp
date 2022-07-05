#include <stdio.h>
#include <omp.h>

int main()
{
    /* 逐次処理のスレッドに関する情報を表示 */
    printf("現在使用中のスレッド数は「%d」です。\n", omp_get_num_threads() );
    printf("使用可能なスレッド数は最大「%d」です。\n", omp_get_max_threads() );

    /* 並列処理を指定した時のスレッドに関する情報を表示 */
    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("現在使用中のスレッド数は「%d」です。\n", omp_get_num_threads() );
            printf("使用可能なスレッド数は最大「%d」です。\n", omp_get_max_threads() );
        }
    }

    /* スレッド数と並列処理を指定した時のスレッドに関する情報を表示 */
    #pragma omp parallel num_threads(10)
    {
        #pragma omp single
        {
            printf("現在使用中のスレッド数は「%d」です。\n", omp_get_num_threads() );
            printf("使用可能なスレッド数は最大「%d」です。\n", omp_get_max_threads() );
        }
    }

    printf( "\n" );
    return 0;
}