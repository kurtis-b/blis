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

#pragma once

#include "scalv.h"
#include "level1/ref_scalv.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for scalv operation.
 */
template<typename T, typename U>
static void test_scalv( char conja_alpha, gtint_t n, gtint_t incx, U alpha, double thresh )
{
    //----------------------------------------------------------
    //        Initialize vector with random numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, n, incx );

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    // Create a copy of y so that we can check reference results.
    std::vector<T> x_ref(x);
    testinghelpers::ref_scalv<T, U>( conja_alpha, n, alpha, x_ref.data(), incx );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    scalv<T, U>( conja_alpha, n, alpha, x.data(), incx );

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<T>( "x", n, x.data(), x_ref.data(), incx, thresh );
}

/**
 * @brief Used to insert Exception Values in x vector.
 */
template<typename T, typename U>
static void test_scalv( char conja_alpha, gtint_t n, gtint_t incx, gtint_t xi,
                        T x_exval, U alpha, double thresh )
{
    //----------------------------------------------------------
    //        Initialize vector with random numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, n, incx );

    // Update the value at index xi to an extreme value, x_exval.
    if ( -1 < xi && xi < n ) x[xi * incx] = x_exval;
    else                     return;

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    // Create a copy of y so that we can check reference results.
    std::vector<T> x_ref(x);
    testinghelpers::ref_scalv<T, U>( conja_alpha, n, alpha, x_ref.data(), incx );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    scalv<T, U>( conja_alpha, n, alpha, x.data(), incx );

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<T>( "x", n, x.data(), x_ref.data(), incx, thresh, true );
}
