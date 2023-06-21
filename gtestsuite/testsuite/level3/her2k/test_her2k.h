/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

#include "her2k.h"
#include "level3/ref_her2k.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>

template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
void test_her2k( char storage, char uplo, char transa, char transb,
    gtint_t m, gtint_t k, gtint_t lda_inc, gtint_t ldb_inc, gtint_t ldc_inc,
    T alpha, RT beta, double thresh )
{
    // Compute the leading dimensions of a, b, and c.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, transa, m, k, lda_inc );
    gtint_t ldb = testinghelpers::get_leading_dimension( storage, transb, m, k, ldb_inc );
    gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', m, m, ldc_inc );

    //----------------------------------------------------------
    //         Initialize matrics with random numbers
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 8, storage, transa, m, k, lda );
    std::vector<T> b = testinghelpers::get_random_matrix<T>( -5, 2, storage, transb, m, k, ldb );
    // Since matrix C, stored in c, is symmetric and we only use the upper or lower
    // part in the computation of her2k and zero-out the rest to ensure
    // that code operates as expected.
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-3, 5, storage, uplo, m, ldc );

    // Create a copy of c so that we can check reference results.
    std::vector<T> c_ref(c);

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    her2k<T>( storage, uplo, transa, transb, m, k, &alpha, a.data(), lda,
                                b.data(), ldb, &beta, c.data(), ldc );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_her2k( storage, uplo, transa, transb, m, k, &alpha,
               a.data(), lda, b.data(), ldb, beta, c_ref.data(), ldc );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( storage, m, m, c.data(), c_ref.data(), ldc, thresh );
}