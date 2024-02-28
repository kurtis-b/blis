/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

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

#include <gtest/gtest.h>
#include "test_subv.h"

class zsubvGenericTest :
        // input params: x or conj(x), vector length, stride size of x, stride size of y
        public ::testing::TestWithParam<std::tuple<char, gtint_t, gtint_t, gtint_t>> {};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zsubvGenericTest);

TEST_P( zsubvGenericTest, FunctionalTest )
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes whether x or conj(x) will be added to y:
    char conj_x = std::get<0>(GetParam());
    // vector length:
    gtint_t n = std::get<1>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<2>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<3>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite subv.h (no netlib version) for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // No adjustment applied yet for complex data.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else
        thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_subv<T>( conj_x, n, incx, incy, thresh );
}

// Prints the test case combination
class zsubvGenericTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,gtint_t,gtint_t,gtint_t>> str) const {
        char conj      = std::get<0>(str.param);
        gtint_t n      = std::get<1>(str.param);
        gtint_t incx   = std::get<2>(str.param);
        gtint_t incy   = std::get<3>(str.param);
        std::string str_name = "bli_zsubv";
        str_name += "_n_" + std::to_string(n);
        str_name += "_conj_" + std::string(&conj, 1);
        std::string incx_str = ( incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name += "_incx_" + incx_str;
        std::string incy_str = ( incy > 0) ? std::to_string(incy) : "m" + std::to_string(std::abs(incy));
        str_name += "_incy_" + incy_str;
        return str_name;
    }
};

#ifdef TEST_BLIS_TYPED
INSTANTIATE_TEST_SUITE_P(
        PositiveIncrements,
        zsubvGenericTest,
        ::testing::Combine(
            // n: use x, c: use conj(x)
            ::testing::Values('n','c'),
            // n: size of vector.
            // as don't have BLIS vectorized kernels for subv,
            // having fewer sizes or maybe a Range would be sufficient
            // to ensure code coverage of the reference kernel.
            ::testing::Values(
                gtint_t( 1),
                gtint_t( 2),
                gtint_t( 3),
                gtint_t( 5),
                gtint_t( 7),
                gtint_t( 9),
                gtint_t(10),
                gtint_t(15),
                gtint_t(20),
                gtint_t(55),
                gtint_t(99)
            ),
            // incx: stride of x vector.
            ::testing::Values(
                gtint_t(1),gtint_t(5)
            ),
            // incy: stride of y vector.
            ::testing::Values(
                gtint_t(1),gtint_t(5)
            )
        ),
        ::zsubvGenericTestPrint()
    );
#endif
