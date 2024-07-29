/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include "blis.h"


// Benchmark application to process aocl logs generated by BLIS library.
#ifndef DT
#define DT BLIS_DOUBLE
#endif

#ifndef IND
#define IND BLIS_NAT
#endif

#ifndef N_REPEAT
//#define N_REPEAT 100
#endif


#define AOCL_MATRIX_INITIALISATION
#define BUFFER_SIZE 256

/* For BLIS since logs are collected at BLAS interfaces
 * we disable cblas interfaces for this benchmark application
 */

#ifdef BLIS_ENABLE_CBLAS
// #define CBLAS
#endif

// #define PRINT

int main( int argc, char** argv )
{
    obj_t a, b, c;
    obj_t c_save;
    obj_t alpha, beta, alpha_one;
    dim_t m, n, k;
    dim_t  p_inc = 0; // to keep track of number of inputs
    num_t dt;
    //    ind_t    ind;
    char     dt_ch;
    int   r, n_repeats;
    trans_t  transa;
    trans_t  transb;

    double   dtime;
    double   dtime_save;
    double   gflops;

    int packA, packB;

    FILE* fin  = NULL;
    FILE* fout = NULL;

    n_repeats = N_REPEAT;  // This macro will get from Makefile.

    dt = DT;

    if (argc < 3)
    {
        printf("Usage: ./test_gemm_pack_compute_XX.x input.csv output.csv\n");
        exit(1);
    }
    fin = fopen(argv[1], "r");
    if (fin == NULL)
    {
        printf("Error opening the file %s\n", argv[1]);
        exit(1);
    }
    fout = fopen(argv[2], "w");
    if (fout == NULL)
    {
        printf("Error opening output file %s\n", argv[2]);
        exit(1);
    }
  if (argc > 3)
  {
    n_repeats = atoi(argv[3]);
  }

    fprintf(fout, "Dt transa transb identifier m n k alphaR alphaI lda ldb betaR betaI ldc gflops\n");

    // Following variables are needed for scanf to read inputs properly
    // however they are not used in bench.
    char api_name[BUFFER_SIZE];       // to store function name, line no present in logs
    char dummy_buffer[BUFFER_SIZE];

    // Variables extracted from the logs which are used by bench
    char stor_scheme, transA_c, transB_c, packA_c, packB_c;
    double alpha_r, beta_r, alpha_i, beta_i;
    dim_t m_trans, n_trans;
    inc_t lda, ldb, ldc;

    stor_scheme = 'C'; // By default set it to Column Major

    //{S, D, C, Z} transa, transb, packA, packB, m, n, k, alpha_real,
    //             alpha_imag, lda ldb, beta_real, beta_imag, ldc,
    //
    //             number of threads, execution time, gflops ---> ignored by bench
    while (fscanf(fin, "%s %c %c %c %c %c " INT_FS INT_FS INT_FS " %lf %lf " INT_FS INT_FS " %lf %lf " INT_FS"[^\n]",
            api_name, &dt_ch, &transA_c, &transB_c, &packA_c, &packB_c, &m, &n, &k, &alpha_r, &alpha_i,
            &lda, &ldb, &beta_r, &beta_i, &ldc) == 16)
    {
        // Discard any extra data on current line in the input file.
        fgets(dummy_buffer, BUFFER_SIZE, fin );

        // At BLAS level only column major order is supported.
        stor_scheme = 'C';

        if (dt_ch == 'D' || dt_ch == 'd') dt = BLIS_DOUBLE;
        else if (dt_ch == 'S' || dt_ch == 's') dt = BLIS_FLOAT;
        else
        {
            printf("Invalid data type %c\n", dt_ch);
            continue;
        }

        if      ( transA_c == 'n' || transA_c == 'N' ) transa = BLIS_NO_TRANSPOSE;
        else if ( transA_c == 't' || transA_c == 'T' ) transa = BLIS_TRANSPOSE;
        else if ( transA_c == 'c' || transA_c == 'C' ) transa = BLIS_CONJ_TRANSPOSE;
        else
        {
            printf("Invalid option for transA \n");
            continue;
        }

        if      ( transB_c == 'n' || transB_c == 'N' ) transb = BLIS_NO_TRANSPOSE;
        else if ( transB_c == 't' || transB_c == 'T' ) transb = BLIS_TRANSPOSE;
        else if ( transB_c == 'c' || transB_c == 'C' ) transb = BLIS_CONJ_TRANSPOSE;
        else
        {
            printf("Invalid option for transB \n");
            continue;
        }

        if      ( packA_c == 'p' || packA_c == 'P' ) packA = TRUE;
        else if ( packA_c == 'u' || packA_c == 'U' ) packA = FALSE;
        else
        {
            printf("Invalid option for packA \n");
            continue;
        }

        if      ( packB_c == 'p' || packB_c == 'P') packB = TRUE;
        else if ( packB_c == 'u' || packB_c == 'U') packB = FALSE;
        else
        {
            printf("Invalid option for packB \n");
            continue;
        }

        bli_obj_create( dt, 1, 1, 0, 0, &alpha);
        bli_obj_create( dt, 1, 1, 0, 0, &beta );

        bli_obj_create( dt, 1, 1, 0, 0, &alpha_one);

        if( (stor_scheme == 'C') || (stor_scheme == 'c') )
        {
            // leading dimension should be greater than number of rows
            // if ((m > lda) || (k > ldb) || (m > ldc)) continue;
            // Since this bench app is run on logs generated by AOCL trace logs
            // - we have relaxed the checks on the input parameters.

            // if A is transpose - A(lda x m), lda >= max(1,k)
            // if A is non-transpose - A (lda x k), lda >= max(1,m)
            // if B is transpose - B (ldb x k), ldb >= max(1,n)
            // if B is non-transpose - B (ldb x n), ldb >= max(1,k)
            //    C is ldc x n - ldc >= max(1, m)
            //if(transa) lda = k; // We will end up overwriting lda
            bli_set_dims_with_trans( transa, m, k, &m_trans, &n_trans);
            bli_obj_create( dt, m_trans, n_trans, 1, lda, &a);

            //if(transb) ldb = n; // we will end up overwriting ldb, ldb >= n
            bli_set_dims_with_trans( transb, k, n, &m_trans, &n_trans);
            bli_obj_create( dt, m_trans, n_trans, 1, ldb, &b);

            bli_obj_create( dt, m, n, 1, ldc, &c);
            bli_obj_create( dt, m, n, 1, ldc, &c_save );
        }
        else if( (stor_scheme == 'r') || (stor_scheme == 'R') )
        {
            //leading dimension should be greater than number of columns
            //if ((k > lda) || (n > ldb) || (n > ldc)) continue;
            // Since this bench app is run on logs generated by AOCL trace logs
            // - we have relaxed the checks on the input parameters.

            // if A is transpose - A(k x lda), lda >= max(1,m)
            // if A is non-transpose - A (m x lda), lda >= max(1,k)
            // if B is transpose - B (n x ldb), ldb >= max(1,k)
            // if B is non-transpose - B (k x ldb ), ldb >= max(1,n)
            //    C is m x ldc - ldc >= max(1, n)

            //if(transa) lda = m; // this will overwrite lda
            bli_set_dims_with_trans(transa, m, k, &m_trans, &n_trans);
            bli_obj_create( dt, m_trans, n_trans, lda, 1, &a);

            //if(transb) ldb = k; // this will overwrite ldb
            bli_set_dims_with_trans(transb, k, n, &m_trans, &n_trans);
            bli_obj_create( dt, m_trans, n_trans, ldb, 1, &b);

            bli_obj_create( dt, m, n, ldc, 1, &c);
            bli_obj_create( dt, m, n, ldc, 1, &c_save );
        }
        else
        {
            printf("Invalid storage scheme\n");
            continue;
        }
#ifndef BLIS // Incase if we are using blis interface we don't have to check for col-storage.
     #ifndef CBLAS
        if( ( stor_scheme == 'R' ) || ( stor_scheme == 'r' ) )
        {
            printf("BLAS APIs doesn't support row-storage: Enable CBLAS\n");
            continue;
        }
     #endif
#endif

#ifdef AOCL_MATRIX_INITIALISATION
        bli_randm( &a );
        bli_randm( &b );
        bli_randm( &c );
#endif
        bli_copym( &c, &c_save );

        bli_obj_set_conjtrans( transa, &a);
        bli_obj_set_conjtrans( transb, &b);

        bli_setsc( 1.0, 1.0, &alpha_one );
        bli_setsc( alpha_r, alpha_i, &alpha );
        bli_setsc( beta_r, beta_i, &beta );

        dtime_save = DBL_MAX;

        for ( r = 0; r < n_repeats; ++r )
        {
            bli_copym( &c_save, &c );
#ifdef PRINT
            bli_printm( "a", &a, "%4.6f", "" );
            bli_printm( "b", &b, "%4.6f", "" );
            bli_printm( "c", &c, "%4.6f", "" );
#endif

#ifdef BLIS

            printf( "BLAS Extension APIs don't have a BLIS interface."
                    "Enable CBLAS or BLAS interface!\n" );

#else

#ifdef CBLAS
            enum CBLAS_ORDER      cblas_order;
            enum CBLAS_TRANSPOSE  cblas_transa;
            enum CBLAS_TRANSPOSE  cblas_transb;
            enum CBLAS_IDENTIFIER cblas_identifierA;
            enum CBLAS_IDENTIFIER cblas_identifierB;

            size_t bufSizeA;
            size_t bufSizeB;

            if ( ( stor_scheme == 'C' ) || ( stor_scheme == 'c' ) )
              cblas_order = CblasColMajor;
            else
              cblas_order = CblasRowMajor;

            if( bli_is_trans( transa ) )
              cblas_transa = CblasTrans;
            else if( bli_is_conjtrans( transa ) )
              cblas_transa = CblasConjTrans;
            else
              cblas_transa = CblasNoTrans;

            if( bli_is_trans( transb ) )
              cblas_transb = CblasTrans;
            else if( bli_is_conjtrans( transb ) )
              cblas_transb = CblasConjTrans;
            else
              cblas_transb = CblasNoTrans;

            if ( packA )
              cblas_identifierA = CblasAMatrix;

            if ( packB )
              cblas_identifierB = CblasBMatrix;
#else
            f77_char f77_transa;
            f77_char f77_transb;
            f77_char f77_identifierA;
            f77_char f77_identifierB;
            f77_int  f77_bufSizeA;
            f77_int  f77_bufSizeB;

            f77_char f77_packed = 'P';
            f77_identifierA = 'A';
            f77_identifierB = 'B';
            bli_param_map_blis_to_netlib_trans( transa, &f77_transa );
            bli_param_map_blis_to_netlib_trans( transb, &f77_transb );

            err_t err = BLIS_SUCCESS;

#endif
            if ( bli_is_float( dt ) )
            {
                f77_int  mm     = bli_obj_length( &c );
                f77_int  kk     = bli_obj_width_after_trans( &a );
                f77_int  nn     = bli_obj_width( &c );

                float*   alphaonep = bli_obj_buffer( &alpha_one );
                float*   alphap = bli_obj_buffer( &alpha );
                float*   ap     = bli_obj_buffer( &a );
                float*   bp     = bli_obj_buffer( &b );
                float*   betap  = bli_obj_buffer( &beta );
                float*   cp     = bli_obj_buffer( &c );

#ifdef CBLAS
                float* aBuffer;
                float* bBuffer;

                if ( packA && !packB )
                {
                  // Only A is pre-packed.
                  bufSizeA = cblas_sgemm_pack_get_size( CblasAMatrix,
                                                        mm,
                                                        nn,
                                                        kk );
                  aBuffer = (float*) bli_malloc_user( bufSizeA, &err );

                  cblas_sgemm_pack( cblas_order,
                                    CblasAMatrix,
                                    cblas_transa,
                                    mm,
                                    nn,
                                    kk,
                                    *alphap,
                                    ap, lda,
                                    aBuffer );

                  dtime = bli_clock();

                  cblas_sgemm_compute( cblas_order,
                                       CblasPacked,
                                       cblas_transb,
                                       mm,
                                       nn,
                                       kk,
                                       aBuffer, lda,
                                       bp, ldb,
                                       *betap,
                                       cp, ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );

                  bli_free_user(aBuffer);
                }
                else if ( !packA && packB )
                {
                  // Only B is pre-packed.
                  bufSizeB = cblas_sgemm_pack_get_size( CblasBMatrix,
                                                        mm,
                                                        nn,
                                                        kk );
                  bBuffer = (float*) bli_malloc_user( bufSizeB, &err );

                  cblas_sgemm_pack( cblas_order,
                                    CblasBMatrix,
                                    cblas_transb,
                                    mm,
                                    nn,
                                    kk,
                                    *alphap,
                                    bp, ldb,
                                    bBuffer );

                  dtime = bli_clock();

                  cblas_sgemm_compute( cblas_order,
                                       cblas_transa,
                                       CblasPacked,
                                       mm,
                                       nn,
                                       kk,
                                       ap, lda,
                                       bBuffer, ldb,
                                       *betap,
                                       cp, ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );


                  bli_free_user(bBuffer);
                }
                else if ( packA && packB )
                {
                  // Both A & B are pre-packed.
                  bufSizeA = cblas_sgemm_pack_get_size( CblasAMatrix,
                                                        mm,
                                                        nn,
                                                        kk );
                  aBuffer = (float*) bli_malloc_user( bufSizeA, &err );

                  bufSizeB = cblas_sgemm_pack_get_size( CblasBMatrix,
                                                        mm,
                                                        nn,
                                                        kk );
                  bBuffer = (float*) bli_malloc_user( bufSizeB, &err );

                  cblas_sgemm_pack( cblas_order,
                                    CblasAMatrix,
                                    cblas_transa,
                                    mm,
                                    nn,
                                    kk,
                                    *alphap,
                                    ap, lda,
                                    aBuffer );

                  cblas_sgemm_pack( cblas_order,
                                    CblasBMatrix,
                                    cblas_transb,
                                    mm,
                                    nn,
                                    kk,
                                    *alphaonep,
                                    bp, ldb,
                                    bBuffer );

                  dtime = bli_clock();

                  cblas_sgemm_compute( cblas_order,
                                       CblasPacked,
                                       CblasPacked,
                                       mm,
                                       nn,
                                       kk,
                                       aBuffer, lda,
                                       bBuffer, ldb,
                                       *betap,
                                       cp, ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );

                  bli_free_user(aBuffer);
                  bli_free_user(bBuffer);
                }
                else
                {
                  // Neither A nor B is pre-packed.

                  dtime = bli_clock();

                  cblas_sgemm_compute( cblas_order,
                                       cblas_transa,
                                       cblas_transb,
                                       mm,
                                       nn,
                                       kk,
                                       ap, lda,
                                       bp, ldb,
                                       *betap,
                                       cp, ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );
                }
#else           // -- BLAS API --
                float* aBuffer;
                float* bBuffer;

                if ( packA && !packB )
                {
                  // Only A is pre-packed.
                  f77_bufSizeA = sgemm_pack_get_size_( &f77_identifierA,
                                                       &mm,
                                                       &nn,
                                                       &kk );
                  aBuffer = (float*) bli_malloc_user( f77_bufSizeA, &err );

                  sgemm_pack_( &f77_identifierA,
                               &f77_transa,
                               &mm,
                               &nn,
                               &kk,
                               alphap,
                               ap,
                               (f77_int*)&lda,
                               aBuffer );

                  dtime = bli_clock();

                  sgemm_compute_( &f77_packed,
                                  &f77_transb,
                                  &mm,
                                  &nn,
                                  &kk,
                                  aBuffer, (f77_int*)&lda,
                                  bp, (f77_int*)&ldb,
                                  betap,
                                  cp, (f77_int*)&ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );

                  bli_free_user( aBuffer );
                }
                else if ( !packA && packB )
                {
                  // Only B is pre-packed.
                  f77_bufSizeB = sgemm_pack_get_size_( &f77_identifierB,
                                                       &mm,
                                                       &nn,
                                                       &kk );
                  bBuffer = (float*) bli_malloc_user( f77_bufSizeB, &err );

                  sgemm_pack_( &f77_identifierB,
                               &f77_transb,
                               &mm,
                               &nn,
                               &kk,
                               alphap,
                               bp,
                               (f77_int*)&ldb,
                               bBuffer );

                  dtime = bli_clock();

                  sgemm_compute_( &f77_transa,
                                  &f77_packed,
                                  &mm,
                                  &nn,
                                  &kk,
                                  ap, (f77_int*)&lda,
                                  bBuffer, (f77_int*)&ldb,
                                  betap,
                                  cp, (f77_int*)&ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );

                  bli_free_user( bBuffer );
                }
                else if ( packA && packB )
                {
                  // Both A & B are pre-packed.
                  f77_bufSizeB = sgemm_pack_get_size_( &f77_identifierB,
                                                       &mm,
                                                       &nn,
                                                       &kk );

                  bBuffer = (float*) bli_malloc_user( f77_bufSizeB, &err );

                  f77_bufSizeA = sgemm_pack_get_size_( &f77_identifierA,
                                                       &mm,
                                                       &nn,
                                                       &kk );

                  aBuffer = (float*) bli_malloc_user( f77_bufSizeA, &err );

                  sgemm_pack_( &f77_identifierA,
                               &f77_transa,
                               &mm,
                               &nn,
                               &kk,
                               alphap,
                               ap,
                               (f77_int*)&lda,
                               aBuffer );

                  sgemm_pack_( &f77_identifierB,
                               &f77_transb,
                               &mm,
                               &nn,
                               &kk,
                               alphaonep,
                               bp,
                               (f77_int*)&ldb,
                               bBuffer );

                  dtime = bli_clock();

                  sgemm_compute_( &f77_packed,
                                  &f77_packed,
                                  &mm,
                                  &nn,
                                  &kk,
                                  aBuffer, (f77_int*)&lda,
                                  bBuffer, (f77_int*)&ldb,
                                  betap,
                                  cp, (f77_int*)&ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );

                  bli_free_user(aBuffer);
                  bli_free_user(bBuffer);
                }
                else
                {
                  // Neither A nor B is reordered.

                  dtime = bli_clock();

                  sgemm_compute_( &f77_transa,
                                  &f77_transb,
                                  &mm,
                                  &nn,
                                  &kk,
                                  ap, (f77_int*)&lda,
                                  bp, (f77_int*)&ldb,
                                  betap,
                                  cp, (f77_int*)&ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );
                }
#endif
            }
            else if ( bli_is_double( dt ) )
            {
                f77_int  mm     = bli_obj_length( &c );
                f77_int  kk     = bli_obj_width_after_trans( &a );
                f77_int  nn     = bli_obj_width( &c );

                double*  alphap = bli_obj_buffer( &alpha );
                double*  alphaonep = bli_obj_buffer( &alpha_one );
                double*  ap     = bli_obj_buffer( &a );
                double*  bp     = bli_obj_buffer( &b );
                double*  betap  = bli_obj_buffer( &beta );
                double*  cp     = bli_obj_buffer( &c );

#ifdef CBLAS
                double* aBuffer;
                double* bBuffer;

                if ( packA && !packB )
                {
                  // Only A is pre-packed.
                  bufSizeA = cblas_dgemm_pack_get_size( CblasAMatrix,
                                                        mm,
                                                        nn,
                                                        kk );
                  aBuffer = (double*) bli_malloc_user( bufSizeA, &err );

                  cblas_dgemm_pack( cblas_order,
                                    CblasAMatrix,
                                    cblas_transa,
                                    mm,
                                    nn,
                                    kk,
                                    *alphap,
                                    ap, lda,
                                    aBuffer );

                  dtime = bli_clock();

                  cblas_dgemm_compute( cblas_order,
                                       CblasPacked,
                                       cblas_transb,
                                       mm,
                                       nn,
                                       kk,
                                       aBuffer, lda,
                                       bp, ldb,
                                       *betap,
                                       cp, ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );

                  bli_free_user(aBuffer);
                }
                else if ( !packA && packB )
                {
                  // Only B is pre-packed.
                  bufSizeB = cblas_dgemm_pack_get_size( CblasBMatrix,
                                                        mm,
                                                        nn,
                                                        kk );

                  cblas_dgemm_pack( cblas_order,
                                    CblasBMatrix,
                                    cblas_transb,
                                    mm,
                                    nn,
                                    kk,
                                    *alphap,
                                    bp, ldb,
                                    bBuffer );

                  dtime = bli_clock();

                  cblas_dgemm_compute( cblas_order,
                                       cblas_transa,
                                       CblasPacked,
                                       mm,
                                       nn,
                                       kk,
                                       ap, lda,
                                       bBuffer, ldb,
                                       *betap,
                                       cp, ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );

                  bli_free_user(bBuffer);
                }
                else if ( packA && packB )
                {
                  // Both A & B are pre-packed.
                  bufSizeA = cblas_dgemm_pack_get_size( CblasAMatrix,
                                                        mm,
                                                        nn,
                                                        kk );
                  aBuffer = (double*) bli_malloc_user( bufSizeA, &err );

                  bufSizeB = cblas_dgemm_pack_get_size( CblasBMatrix,
                                                        mm,
                                                        nn,
                                                        kk );
                  bBuffer = (double*) bli_malloc_user( bufSizeB, &err );

                  cblas_dgemm_pack( cblas_order,
                                    CblasAMatrix,
                                    cblas_transa,
                                    mm,
                                    nn,
                                    kk,
                                    *alphap,
                                    ap, lda,
                                    aBuffer );

                  cblas_dgemm_pack( cblas_order,
                                    CblasBMatrix,
                                    cblas_transb,
                                    mm,
                                    nn,
                                    kk,
                                    *alphap,
                                    bp, ldb,
                                    bBuffer );

                  dtime = bli_clock();

                  cblas_dgemm_compute( cblas_order,
                                       CblasPacked,
                                       CblasPacked,
                                       mm,
                                       nn,
                                       kk,
                                       aBuffer, lda,
                                       bBuffer, ldb,
                                       *betap,
                                       cp, ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );

                  bli_free_user(aBuffer);
                  bli_free_user(bBuffer);
                }
                else
                {
                  // Neither A nor B is pre-packed.

                  dtime = bli_clock();

                  cblas_dgemm_compute( cblas_order,
                                       cblas_transa,
                                       cblas_transb,
                                       mm,
                                       nn,
                                       kk,
                                       ap, lda,
                                       bp, ldb,
                                       *betap,
                                       cp, ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );
                }

#else           // -- BLAS API --
                double* aBuffer;
                double* bBuffer;

                if ( packA && !packB )
                {
                  // Only A is pre-packed.
                  f77_bufSizeA = dgemm_pack_get_size_( &f77_identifierA,
                                                       &mm,
                                                       &nn,
                                                       &kk );
                  aBuffer = (double*) bli_malloc_user( f77_bufSizeA, &err );

                  dgemm_pack_( &f77_identifierA,
                               &f77_transa,
                               &mm,
                               &nn,
                               &kk,
                               alphap,
                               ap,
                               (f77_int*)&lda,
                               aBuffer );

                  dtime = bli_clock();

                  dgemm_compute_( &f77_packed,
                                  &f77_transb,
                                  &mm,
                                  &nn,
                                  &kk,
                                  aBuffer, (f77_int*)&lda,
                                  bp, (f77_int*)&ldb,
                                  betap,
                                  cp, (f77_int*)&ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );

                  bli_free_user( aBuffer );
                }
                else if ( !packA && packB )
                {
                  // Only B is pre-packed.
                  f77_bufSizeB = dgemm_pack_get_size_( &f77_identifierB,
                                                       &mm,
                                                       &nn,
                                                       &kk );
                  bBuffer = (double*) bli_malloc_user( f77_bufSizeB, &err );

                  dgemm_pack_( &f77_identifierB,
                               &f77_transb,
                               &mm,
                               &nn,
                               &kk,
                               alphap,
                               bp,
                               (f77_int*)&ldb,
                               bBuffer );

                  dtime = bli_clock();

                  dgemm_compute_( &f77_transa,
                                  &f77_packed,
                                  &mm,
                                  &nn,
                                  &kk,
                                  ap, (f77_int*)&lda,
                                  bBuffer, (f77_int*)&ldb,
                                  betap,
                                  cp, (f77_int*)&ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );

                  bli_free_user( bBuffer );
                }
                else if ( packA && packB )
                {
                  // Both A & B are pre-packed.
                  f77_bufSizeA = dgemm_pack_get_size_( &f77_identifierA,
                                                       &mm,
                                                       &nn,
                                                       &kk );
                  aBuffer = (double*) bli_malloc_user( f77_bufSizeA, &err );

                  f77_bufSizeB = dgemm_pack_get_size_( &f77_identifierB,
                                                       &mm,
                                                       &nn,
                                                       &kk );
                  bBuffer = (double*) bli_malloc_user( f77_bufSizeB, &err );

                  dgemm_pack_( &f77_identifierA,
                               &f77_transa,
                               &mm,
                               &nn,
                               &kk,
                               alphap,
                               ap,
                               (f77_int*)&lda,
                               aBuffer );

                  dgemm_pack_( &f77_identifierB,
                               &f77_transb,
                               &mm,
                               &nn,
                               &kk,
                               alphaonep,
                               bp,
                               (f77_int*)&ldb,
                               bBuffer );

                  dtime = bli_clock();

                  dgemm_compute_( &f77_packed,
                                  &f77_packed,
                                  &mm,
                                  &nn,
                                  &kk,
                                  aBuffer, (f77_int*)&lda,
                                  bBuffer, (f77_int*)&ldb,
                                  betap,
                                  cp, (f77_int*)&ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );

                  bli_free_user(aBuffer);
                  bli_free_user(bBuffer);
                }
                else
                {
                  // Neither A nor B is reordered.

                  dtime = bli_clock();

                  dgemm_compute_( &f77_transa,
                                  &f77_transb,
                                  &mm,
                                  &nn,
                                  &kk,
                                  ap, (f77_int*)&lda,
                                  bp, (f77_int*)&ldb,
                                  betap,
                                  cp, (f77_int*)&ldc );

                  dtime_save = bli_clock_min_diff( dtime_save, dtime );
                }
#endif
            }
#endif

#ifdef PRINT
            bli_printm( "c compute", &c, "%4.6f", "" );
#endif
        }

        gflops = ( 2.0 * m * k * n ) / ( dtime_save * 1.0e9 );

        if ( bli_is_complex( dt ) ) gflops *= 4.0;

        printf( "data_%cgemm_%s", dt_ch, BLAS );

        p_inc++;
        printf("( %2lu, 1:4 ) = [ %4lu %4lu %4lu %7.2f ];\n",
               (unsigned long)(p_inc),
               (unsigned long)m,
               (unsigned long)n,
               (unsigned long)k, gflops);

        fprintf (fout, "%c %c %c %c %c %ld %ld %ld %lf %lf %ld %ld %lf %lf %ld %6.3f\n", \
                 dt_ch, transA_c, transB_c, packA_c, packB_c, m, n, k, alpha_r, alpha_i, lda, ldb, beta_r, beta_i, ldc, gflops);

        fflush(fout);

        bli_obj_free( &alpha );
        bli_obj_free( &beta );

        bli_obj_free( &a );
        bli_obj_free( &b );
        bli_obj_free( &c );
        bli_obj_free( &c_save );
    }

    //bli_finalize();
    fclose(fin);
    fclose(fout);

    return 0;
}
