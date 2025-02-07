/*
$gcc test_lpgemm.c -o ./test_lpgemm.x -I/aocl-blis_install_directory/include/amdzen/
-L/aocl-blis_install_directory/lib/amdzen/ -lblis-mt -lm

Note: Export blis library path to LD_LIBRARY_PATH before running the
executable ./test_lpgem.x
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "blis.h"

// Example program to demonstrate LPGEMM API usage.
// aocl_gemm_f32f32f32of32 (A:float, B:float, C:float) used here.
int main()
{
    dim_t m = 512;
    dim_t n = 4096;
    dim_t k = 4096;

    // Leading dimensions for row major matrices.
    dim_t lda = k;
    dim_t ldb = n;
    dim_t ldc = n;
    inc_t r, n_repeats;
    n_repeats = 300;

    double dtime;
    double dtime_save;
    double gflops;

    err_t err = BLIS_SUCCESS;
    int8_t *a = (int8_t *)bli_malloc_user(sizeof(int8_t) * m * k, &err);
    if (err != BLIS_SUCCESS)
    {
        goto bailout;
    }

    int8_t *b = (int8_t *)bli_malloc_user(sizeof(int8_t) * n * k, &err);
    if (err != BLIS_SUCCESS)
    {
        goto bailout;
    }

    int16_t *c = (int16_t *)bli_malloc_user(sizeof(int16_t) * m * n, &err);
    if (err != BLIS_SUCCESS)
    {
        goto bailout;
    }

    // Seed the random number generator
    srand(time(NULL));

    // Fill matrix a with random values
    for (dim_t i = 0; i < m * k; i++)
    {
        a[i] = rand() % 256 - 128; // Random values between -128 and 127
    }

    // Fill matrix b with random values
    for (dim_t i = 0; i < n * k; i++)
    {
        b[i] = rand() % 256 - 128; // Random values between -128 and 127
    }

    // Functions to fill the matrices with data can be added here.
    float alpha = 1.0;
    float beta = 1.0;
    char storage = 'r'; // Row major. Use 'c' for column major.
    char transa = 'n';  // No transpose. Transpose not supported for all API's.
    char transb = 'n';
    char reordera = 'n';
    char reorderb = 'n';

    dtime_save = DBL_MAX;
    for (r = 0; r < n_repeats; ++r)
    {
        dtime = bli_clock();
        aocl_gemm_s8s8s16os16(storage, transa, transb,
                              m, n, k,
                              alpha,
                              a, lda, reordera,
                              b, ldb, reorderb,
                              beta,
                              c, ldc,
                              NULL);
        dtime_save = bli_clock_min_diff(dtime_save, dtime);
    } // nrepeats

    gflops = (2.0 * m * k * n) / (dtime_save * 1.0e9);
    printf("~~~~~~~~~~_LPGEMM in AOCL-BLAS\n");
    printf("m\t k\t n\t cs_a\t cs_b\t cs_c \t gops\n");
    printf("%lu \t %4lu \t %4lu \t %4lu \t %4lu \t %4lu \t %6.3f\n",
           (unsigned long)m, (unsigned long)k, (unsigned long)n,
           (unsigned long)lda, (unsigned long)ldb, (unsigned long)ldc, gflops);

bailout:

    if (a != NULL)
    {
        bli_free_user(a);
    }
    if (b != NULL)
    {
        bli_free_user(b);
    }
    if (c != NULL)
    {
        bli_free_user(c);
    }

    return 0;
}
