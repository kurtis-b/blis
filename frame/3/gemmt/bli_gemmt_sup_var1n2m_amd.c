/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#include "blis.h"

#define FUNCPTR_T gemmtsup_fp

typedef void (*FUNCPTR_T)
     (
       bool             packa,
       bool             packb,
       conj_t           conja,
       conj_t           conjb,
       dim_t            m,
       dim_t            n,
       dim_t            k,
       void*   restrict alpha,
       void*   restrict a, inc_t rs_a, inc_t cs_a,
       void*   restrict b, inc_t rs_b, inc_t cs_b,
       void*   restrict beta,
       void*   restrict c, inc_t rs_c, inc_t cs_c,
       stor3_t          eff_id,
       cntx_t* restrict cntx,
       rntm_t* restrict rntm,
       thrinfo_t* restrict thread
     );


// Declaration of gemmt specific kernels function pointer
// This is aligned to bli_dgemmsup_rv_haswell_asm_6x8m function protype.
typedef void (*gemmt_ker_ft)
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     );

// these kernels are compiled as part of zen4 config
// use them only when BLIS_KERNELS_ZEN4 is defined
// Look-up table for Gemmt Upper Variant Kernels
#if defined(BLIS_KERNELS_ZEN4)
gemmt_ker_ft ker_fpus_zen4[3] =
	{
		bli_dgemmsup_rv_zen4_asm_24x8m_upper_0,
		bli_dgemmsup_rv_zen4_asm_24x8m_upper_1,
		bli_dgemmsup_rv_zen4_asm_24x8m_upper_2
	};

//Look-up table for Gemmt Lower Variant Kernels
gemmt_ker_ft ker_fpls_zen4[3] = 
	{
		bli_dgemmsup_rv_zen4_asm_24x8m_lower_0,
		bli_dgemmsup_rv_zen4_asm_24x8m_lower_1,
		bli_dgemmsup_rv_zen4_asm_24x8m_lower_2
	};
#endif 

// these kernels are compiled as part of haswell config
// use them only when BLIS_KERNELS_HASWELL is defined
#if defined(BLIS_KERNELS_HASWELL)
//Look-up table for Gemmt Upper Variant Kernels
gemmt_ker_ft ker_fpus_haswell[14] =
	{
		bli_dgemmsup_rv_haswell_asm_6x8m_0x0_U,
		bli_dgemmsup_rv_haswell_asm_6x8m_6x0_U,
		bli_dgemmsup_rv_haswell_asm_6x8m_6x8_U,
		bli_dgemmsup_rv_haswell_asm_6x8m_12x8_U,
		bli_dgemmsup_rv_haswell_asm_6x8m_12x16_U,
		bli_dgemmsup_rv_haswell_asm_6x8m_18x16_U,
		bli_dgemmsup_rv_haswell_asm_6x8m_0x0_combined_U,
		bli_dgemmsup_rd_haswell_asm_6x8m_0x0_U,
		bli_dgemmsup_rd_haswell_asm_6x8m_6x0_U,
		bli_dgemmsup_rd_haswell_asm_6x8m_6x8_U,
		bli_dgemmsup_rd_haswell_asm_6x8m_12x8_U,
		bli_dgemmsup_rd_haswell_asm_6x8m_12x16_U,
		bli_dgemmsup_rd_haswell_asm_6x8m_18x16_U,
		bli_dgemmsup_rd_haswell_asm_6x8m_0x0_combined_U};

//Look-up table for Gemmt Lower Variant Kernels
gemmt_ker_ft ker_fpls_haswell[14] = 
	{
		bli_dgemmsup_rv_haswell_asm_6x8m_0x0_L,
		bli_dgemmsup_rv_haswell_asm_6x8m_6x0_L,
		bli_dgemmsup_rv_haswell_asm_6x8m_6x8_L,
		bli_dgemmsup_rv_haswell_asm_6x8m_12x8_L,
		bli_dgemmsup_rv_haswell_asm_6x8m_12x16_L,
		bli_dgemmsup_rv_haswell_asm_6x8m_18x16_L,
		bli_dgemmsup_rv_haswell_asm_6x8m_16x12_combined_L,
		bli_dgemmsup_rd_haswell_asm_6x8m_0x0_L,
		bli_dgemmsup_rd_haswell_asm_6x8m_6x0_L,
		bli_dgemmsup_rd_haswell_asm_6x8m_6x8_L,
		bli_dgemmsup_rd_haswell_asm_6x8m_12x8_L,
		bli_dgemmsup_rd_haswell_asm_6x8m_12x16_L,
		bli_dgemmsup_rd_haswell_asm_6x8m_18x16_L,
		bli_dgemmsup_rd_haswell_asm_6x8m_16x12_combined_L
	};
#else
gemmt_ker_ft ker_fpls[1];
gemmt_ker_ft ker_fpus[1];
#endif

//
// -- var1n --------------------------------------------------------------------
//

static FUNCPTR_T GENARRAY_T(ftypes_var1n,gemmtsup,ref_var1n);

void bli_gemmtsup_ref_var1n
     (
       trans_t trans,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       stor3_t eff_id,
       cntx_t* cntx,
       rntm_t* rntm,
       thrinfo_t* thread
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_5);
#if 0
	obj_t at, bt;

	bli_obj_alias_to( a, &at );
	bli_obj_alias_to( b, &bt );

	// Induce transpositions on A and/or B if either object is marked for
	// transposition. We can induce "fast" transpositions since they objects
	// are guaranteed to not have structure or be packed.
	if ( bli_obj_has_trans( &at ) ) { bli_obj_induce_fast_trans( &at ); }
	if ( bli_obj_has_trans( &bt ) ) { bli_obj_induce_fast_trans( &bt ); }

	const num_t    dt        = bli_obj_dt( c );

	const conj_t   conja     = bli_obj_conj_status( a );
	const conj_t   conjb     = bli_obj_conj_status( b );

	const dim_t    m         = bli_obj_length( c );
	const dim_t    n         = bli_obj_width( c );

	const dim_t    k         = bli_obj_width( &at );

	void* restrict buf_a     = bli_obj_buffer_at_off( &at );
	const inc_t    rs_a      = bli_obj_row_stride( &at );
	const inc_t    cs_a      = bli_obj_col_stride( &at );

	void* restrict buf_b     = bli_obj_buffer_at_off( &bt );
	const inc_t    rs_b      = bli_obj_row_stride( &bt );
	const inc_t    cs_b      = bli_obj_col_stride( &bt );

	void* restrict buf_c     = bli_obj_buffer_at_off( c );
	const inc_t    rs_c      = bli_obj_row_stride( c );
	const inc_t    cs_c      = bli_obj_col_stride( c );

	void* restrict buf_alpha = bli_obj_buffer_for_1x1( dt, alpha );
	void* restrict buf_beta  = bli_obj_buffer_for_1x1( dt, beta );

#else
	const num_t    dt        = bli_obj_dt( c );

	const bool     packa     = bli_rntm_pack_a( rntm );
	const bool     packb     = bli_rntm_pack_b( rntm );

	const conj_t   conja     = bli_obj_conj_status( a );
	const conj_t   conjb     = bli_obj_conj_status( b );

	const dim_t    m         = bli_obj_length( c );
	const dim_t    n         = bli_obj_width( c );
	      dim_t    k;

	void* restrict buf_a = bli_obj_buffer_at_off( a );
	      inc_t    rs_a;
	      inc_t    cs_a;

	void* restrict buf_b = bli_obj_buffer_at_off( b );
	      inc_t    rs_b;
	      inc_t    cs_b;

	if ( bli_obj_has_notrans( a ) )
	{
		k     = bli_obj_width( a );

		rs_a  = bli_obj_row_stride( a );
		cs_a  = bli_obj_col_stride( a );
	}
	else // if ( bli_obj_has_trans( a ) )
	{
		// Assign the variables with an implicit transposition.
		k     = bli_obj_length( a );

		rs_a  = bli_obj_col_stride( a );
		cs_a  = bli_obj_row_stride( a );
	}

	if ( bli_obj_has_notrans( b ) )
	{
		rs_b  = bli_obj_row_stride( b );
		cs_b  = bli_obj_col_stride( b );
	}
	else // if ( bli_obj_has_trans( b ) )
	{
		// Assign the variables with an implicit transposition.
		rs_b  = bli_obj_col_stride( b );
		cs_b  = bli_obj_row_stride( b );
	}


	// Optimize some storage/packing cases by transforming them into others.
	// These optimizations are expressed by changing trans and/or eff_id.
	bli_gemmsup_ref_var1n2m_opt_cases( dt, &trans, packa, packb, &eff_id, cntx );


	bool uploc;
	if( bli_obj_is_lower( c ) )
	{
		uploc = 0;
	}
	else
	{
		uploc = 1;
	}

	void* restrict buf_c     = bli_obj_buffer_at_off( c );
	const inc_t    rs_c      = bli_obj_row_stride( c );
	const inc_t    cs_c      = bli_obj_col_stride( c );

	void* restrict buf_alpha = bli_obj_buffer_for_1x1( dt, alpha );
	void* restrict buf_beta  = bli_obj_buffer_for_1x1( dt, beta );

#endif

	// Index into the type combination array to extract the correct
	// function pointer.
	FUNCPTR_T f = ftypes_var1n[dt][uploc];

	if ( bli_is_notrans( trans ) )
	{
		// Invoke the function.
		f
		(
		  packa,
		  packb,
		  conja,
		  conjb,
		  m,
		  n,
		  k,
		  buf_alpha,
		  buf_a, rs_a, cs_a,
		  buf_b, rs_b, cs_b,
		  buf_beta,
		  buf_c, rs_c, cs_c,
		  eff_id,
		  cntx,
		  rntm,
		  thread
		);
	}
	else
	{
		// Invoke the function (transposing the operation).
		f
		(
		  packb,
		  packa,
		  conjb,             // swap the conj values.
		  conja,
		  n,                 // swap the m and n dimensions.
		  m,
		  k,
		  buf_alpha,
		  buf_b, cs_b, rs_b, // swap the positions of A and B.
		  buf_a, cs_a, rs_a, // swap the strides of A and B.
		  buf_beta,
		  buf_c, cs_c, rs_c, // swap the strides of C.
		  bli_stor3_trans( eff_id ), // transpose the stor3_t id.
		  cntx,
		  rntm,
		  thread
		);
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_5);
}

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, uplo, varname ) \
\
void PASTEMACT(ch,opname,uplo,varname) \
     ( \
       bool             packa, \
       bool             packb, \
       conj_t           conja, \
       conj_t           conjb, \
       dim_t            m, \
       dim_t            n, \
       dim_t            k, \
       void*   restrict alpha, \
       void*   restrict a, inc_t rs_a, inc_t cs_a, \
       void*   restrict b, inc_t rs_b, inc_t cs_b, \
       void*   restrict beta, \
       void*   restrict c, inc_t rs_c, inc_t cs_c, \
       stor3_t          stor_id, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	/* If m or n is zero, return immediately. */ \
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* If k < 1 or alpha is zero, scale by beta and return. */ \
	if ( k < 1 || PASTEMAC(ch,eq0)( *(( ctype* )alpha) ) ) \
	{ \
		if ( bli_thread_am_ochief( thread ) ) \
		{ \
			PASTEMAC(ch,scalm) \
			( \
			  BLIS_NO_CONJUGATE, \
			  0, \
			  BLIS_NONUNIT_DIAG, \
			  BLIS_DENSE, \
			  m, n, \
			  beta, \
			  c, rs_c, cs_c \
			); \
		} \
		return; \
	} \
\
	/* This transposition of the stor3_t id value is inherent to variant 1.
	   The reason: we assume that variant 2 is the "main" variant. The
	   consequence of this is that we assume that the millikernels that
	   iterate over m are registered to the "primary" kernel group associated
	   with the kernel IO preference; similarly, mkernels that iterate over
	   n are assumed to be registered to the "non-primary" group associated
	   with the ("non-primary") anti-preference. Note that this pattern holds
	   regardless of whether the mkernel set has a row or column preference.)
	   See bli_l3_sup_int.c for a higher-level view of how this choice is made. */ \
	stor_id = bli_stor3_trans( stor_id ); \
\
	/* Query the context for various blocksizes. */ \
	dim_t NR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NR, cntx ); \
	dim_t MR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MR, cntx ); \
	dim_t NC0 = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NC, cntx ); \
	dim_t MC0 = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MC, cntx ); \
	dim_t KC0 = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_KC, cntx ); \
	/* Query the maximum blocksize for MR, which implies a maximum blocksize
	extension for the final iteration. */ \
	dim_t MRM = bli_cntx_get_l3_sup_tri_blksz_max_dt( dt, BLIS_MR, cntx ); \
\
	/* Query the context for the sup microkernel address and cast it to its
	   function pointer type. */ \
	PASTECH(ch,gemmsup_ker_ft) \
               gemmsup_ker = bli_cntx_get_l3_sup_tri_ker_dt( dt, stor_id, cntx ); \
\
	if( ( 0 == NR ) || ( 0 == MR )  || ( 0 == NC0 ) || ( 0 == MC0 ) || ( 0 == KC0 ) ) \
	{ \
		NR = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx ); \
		MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx ); \
		NC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx ); \
		MC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx ); \
		KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx ); \
		MRM = bli_cntx_get_l3_sup_blksz_max_dt( dt, BLIS_MR, cntx ); \
		gemmsup_ker = bli_cntx_get_l3_sup_ker_dt( dt, stor_id, cntx ); \
	} \
	const dim_t MRE = MRM - MR; \
\
	dim_t KC; \
	if      ( packa && packb ) \
	{ \
		KC = KC0; \
	} \
	else if ( packb ) \
	{ \
		if      ( stor_id == BLIS_RRR || \
				  stor_id == BLIS_CCC    ) KC = KC0; \
		else if ( stor_id == BLIS_RRC || \
				  stor_id == BLIS_CRC    ) KC = KC0; \
		else if ( stor_id == BLIS_RCR || \
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4; \
		else                               KC = KC0; \
	} \
	else if ( packa ) \
	{ \
		if      ( stor_id == BLIS_RRR || \
				  stor_id == BLIS_CCC    ) KC = (( KC0 / 2 ) / 2 ) * 2; \
		else if ( stor_id == BLIS_RRC || \
				  stor_id == BLIS_CRC    ) KC = KC0; \
		else if ( stor_id == BLIS_RCR || \
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4; \
		else                               KC = KC0; \
	} \
	else /* if ( !packa && !packb ) */ \
	{ \
		if      ( FALSE                  ) KC = KC0; \
		else if ( stor_id == BLIS_RRC || \
				  stor_id == BLIS_CRC    ) KC = KC0; \
		else if ( m <=   MR && n <=   NR ) KC = KC0; \
		else if ( m <= 2*MR && n <= 2*NR ) KC = KC0 / 2; \
		else if ( m <= 3*MR && n <= 3*NR ) KC = (( KC0 / 3 ) / 4 ) * 4; \
		else if ( m <= 4*MR && n <= 4*NR ) KC = KC0 / 4; \
		else                               KC = (( KC0 / 5 ) / 4 ) * 4; \
	} \
\
	/* Nudge NC up to a multiple of MR and MC up to a multiple of NR.
	   NOTE: This is unique to variant 1 (ie: not performed in variant 2)
	   because MC % MR == 0 and NC % NR == 0 is already enforced at runtime. */ \
	const dim_t NC  = bli_align_dim_to_mult( NC0, MR ); \
	const dim_t MC  = bli_align_dim_to_mult( MC0, NR ); \
\
	/* Compute partitioning step values for each matrix of each loop. */ \
	const inc_t jcstep_c = rs_c; \
	const inc_t jcstep_a = rs_a; \
\
	const inc_t pcstep_a = cs_a; \
	const inc_t pcstep_b = rs_b; \
\
	const inc_t icstep_c = cs_c; \
	const inc_t icstep_b = cs_b; \
\
	const inc_t jrstep_c = rs_c * MR; \
\
	/*
	const inc_t jrstep_a = rs_a * MR; \
\
	const inc_t irstep_c = cs_c * NR; \
	const inc_t irstep_b = cs_b * NR; \
	*/ \
\
	ctype* restrict a_00       = a; \
	ctype* restrict b_00       = b; \
	ctype* restrict c_00       = c; \
	ctype* restrict alpha_cast = alpha; \
	ctype* restrict beta_cast  = beta; \
\
	/* Make local copies of beta and one scalars to prevent any unnecessary
	   sharing of cache lines between the cores' caches. */ \
	ctype           beta_local = *beta_cast; \
	ctype           one_local  = *PASTEMAC(ch,1); \
\
	auxinfo_t       aux; \
\
	/* Parse and interpret the contents of the rntm_t object to properly
	   set the ways of parallelism for each loop. */ \
	/*bli_rntm_set_ways_from_rntm_sup( m, n, k, rntm );*/ \
\
	/* Initialize a mem_t entry for A and B. Strictly speaking, this is only
	   needed for the matrix we will be packing (if any), but we do it
	   unconditionally to be safe. An alternative way of initializing the
	   mem_t entries is:

	     bli_mem_clear( &mem_a ); \
	     bli_mem_clear( &mem_b ); \
	*/ \
	mem_t mem_a = BLIS_MEM_INITIALIZER; \
	mem_t mem_b = BLIS_MEM_INITIALIZER; \
\
	/* Define an array of bszid_t ids, which will act as our substitute for
	   the cntl_t tree.
	   NOTE: These bszid_t values, and their order, match that of the bp
	   algorithm (variant 2) because they are not used to query actual
	   blocksizes but rather query the ways of parallelism for the various
	   loops. For example, the 2nd loop in variant 1 partitions in the m
	   dimension (in increments of MR), but parallelizes that m dimension
	   with BLIS_JR_NT. The only difference is that the _packa and _packb
	   arrays have been adjusted for the semantic difference in order in
	   which packa and packb nodes are encountered in the thrinfo tree.
	   That is, this panel-block algorithm partitions an NC x KC submatrix
	   of A to be packed in the 4th loop, and a KC x MC submatrix of B
	   to be packed in the 3rd loop. */ \
	/*                           5thloop  4thloop         packa  3rdloop         packb  2ndloop  1stloop  ukrloop */ \
	bszid_t bszids_nopack[6] = { BLIS_NC, BLIS_KC,               BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t bszids_packa [7] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t bszids_packb [7] = { BLIS_NC, BLIS_KC,               BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t bszids_packab[8] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t* restrict bszids; \
\
	/* Set the bszids pointer to the correct bszids array above based on which
	   matrices (if any) are being packed. */ \
	if ( packa ) { if ( packb ) bszids = bszids_packab; \
	               else         bszids = bszids_packa; } \
	else         { if ( packb ) bszids = bszids_packb; \
	               else         bszids = bszids_nopack; } \
\
	/* Determine whether we are using more than one thread. */ \
	const bool is_mt = bli_rntm_calc_num_threads( rntm ); \
\
	thrinfo_t* restrict thread_jc = NULL; \
	thrinfo_t* restrict thread_pc = NULL; \
	thrinfo_t* restrict thread_pa = NULL; \
	thrinfo_t* restrict thread_ic = NULL; \
	thrinfo_t* restrict thread_pb = NULL; \
	thrinfo_t* restrict thread_jr = NULL; \
\
	/* Grow the thrinfo_t tree. */ \
	bszid_t*   restrict bszids_jc = bszids; \
	                    thread_jc = thread; \
	bli_thrinfo_sup_grow( rntm, bszids_jc, thread_jc ); \
\
	/* Compute the JC loop thread range for the current thread. */ \
	dim_t jc_start, jc_end; \
	bli_thread_range_sub( thread_jc, m, MR, FALSE, &jc_start, &jc_end ); \
	const dim_t m_local = jc_end - jc_start; \
\
	/* Compute number of primary and leftover components of the JC loop. */ \
	/*const dim_t jc_iter = ( m_local + NC - 1 ) / NC;*/ \
	const dim_t jc_left =   m_local % NC; \
\
	/* Loop over the m dimension (NC rows/columns at a time). */ \
	/*for ( dim_t jj = 0; jj < jc_iter; jj += 1 )*/ \
	for ( dim_t jj = jc_start; jj < jc_end; jj += NC ) \
	{ \
		/* Calculate the thread's current JC block dimension. */ \
		const dim_t nc_cur = ( NC <= jc_end - jj ? NC : jc_left ); \
\
		ctype* restrict a_jc = a_00 + jj * jcstep_a; \
		ctype* restrict c_jc = c_00 + jj * jcstep_c; \
\
		/* Grow the thrinfo_t tree. */ \
		bszid_t*   restrict bszids_pc = &bszids_jc[1]; \
		                    thread_pc = bli_thrinfo_sub_node( thread_jc ); \
		bli_thrinfo_sup_grow( rntm, bszids_pc, thread_pc ); \
\
		/* Compute the PC loop thread range for the current thread. */ \
		const dim_t pc_start = 0, pc_end = k; \
		const dim_t k_local = k; \
\
		/* Compute number of primary and leftover components of the PC loop. */ \
		/*const dim_t pc_iter = ( k_local + KC - 1 ) / KC;*/ \
		const dim_t pc_left =   k_local % KC; \
\
		/* Loop over the k dimension (KC rows/columns at a time). */ \
		/*for ( dim_t pp = 0; pp < pc_iter; pp += 1 )*/ \
		for ( dim_t pp = pc_start; pp < pc_end; pp += KC ) \
		{ \
			/* Calculate the thread's current PC block dimension. */ \
			const dim_t kc_cur = ( KC <= pc_end - pp ? KC : pc_left ); \
\
			ctype* restrict a_pc = a_jc + pp * pcstep_a; \
			ctype* restrict b_pc = b_00 + pp * pcstep_b; \
\
			/* Only apply beta to the first iteration of the pc loop. */ \
			ctype* restrict beta_use = ( pp == 0 ? &beta_local : &one_local ); \
\
			ctype* a_use; \
			inc_t  rs_a_use, cs_a_use, ps_a_use; \
\
			/* Set the bszid_t array and thrinfo_t pointer based on whether
			   we will be packing A. If we won't be packing A, we alias to
			   the _pc variables so that code further down can unconditionally
			   reference the _pa variables. Note that *if* we will be packing
			   A, the thrinfo_t node will have already been created by a
			   previous call to bli_thrinfo_grow(), since bszid values of
			   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
			   bszid that is a normal bszid_t value). */ \
			bszid_t*   restrict bszids_pa; \
			if ( packa ) { bszids_pa = &bszids_pc[1]; \
			               thread_pa = bli_thrinfo_sub_node( thread_pc ); } \
			else         { bszids_pa = &bszids_pc[0]; \
			               thread_pa = thread_pc; } \
\
			/* Determine the packing buffer and related parameters for matrix
			   A. (If A will not be packed, then a_use will be set to point to
			   a and the _a_use strides will be set accordingly.) Then call
			   the packm sup variant chooser, which will call the appropriate
			   implementation based on the schema deduced from the stor_id.
			   NOTE: packing matrix A in this panel-block algorithm corresponds
			   to packing matrix B in the block-panel algorithm. */ \
			PASTEMAC(ch,packm_sup_a) \
			( \
			  packa, \
			  BLIS_BUFFER_FOR_B_PANEL, /* This algorithm packs matrix A to */ \
			  stor_id,                 /* a "panel of B".                  */ \
			  BLIS_NO_TRANSPOSE, \
			  NC,     KC,       /* This "panel of B" is (at most) NC x KC. */ \
			  nc_cur, kc_cur, MR, \
			  &one_local, \
			  a_pc,   rs_a,      cs_a, \
			  &a_use, &rs_a_use, &cs_a_use, \
			                     &ps_a_use, \
			  cntx, \
			  rntm, \
			  &mem_a, \
			  thread_pa  \
			); \
\
			/* Alias a_use so that it's clear this is our current block of
			   matrix A. */ \
			ctype* restrict a_pc_use = a_use; \
\
			/* We don't need to embed the panel stride of A within the auxinfo_t
			   object because this variant iterates through A in the jr loop,
			   which occurs here, within the macrokernel, not within the
			   millikernel. */ \
			/*bli_auxinfo_set_ps_a( ps_a_use, &aux );*/ \
\
			/* Grow the thrinfo_t tree. */ \
			bszid_t*   restrict bszids_ic = &bszids_pa[1]; \
			                    thread_ic = bli_thrinfo_sub_node( thread_pa ); \
			bli_thrinfo_sup_grow( rntm, bszids_ic, thread_ic ); \
\
			/* Compute the IC loop thread range for the current thread. */ \
			dim_t ic_start, ic_end; \
			bli_thread_range_sub( thread_ic, n, NR, FALSE, &ic_start, &ic_end ); \
			const dim_t n_local = ic_end - ic_start; \
\
			/* Compute number of primary and leftover components of the IC loop. */ \
			/*const dim_t ic_iter = ( n_local + MC - 1 ) / MC;*/ \
			const dim_t ic_left =   n_local % MC; \
\
			/* Loop over the n dimension (MC rows at a time). */ \
			/*for ( dim_t ii = 0; ii < ic_iter; ii += 1 )*/ \
			for ( dim_t ii = ic_start; ii < ic_end; ii += MC ) \
			{ \
				/* Calculate the thread's current IC block dimension. */ \
				const dim_t mc_cur = ( MC <= ic_end - ii ? MC : ic_left ); \
\
				ctype* restrict b_ic = b_pc + ii * icstep_b; \
				ctype* restrict c_ic = c_jc + ii * icstep_c; \
\
				ctype* b_use; \
				inc_t  rs_b_use, cs_b_use, ps_b_use; \
\
				/* Set the bszid_t array and thrinfo_t pointer based on whether
				   we will be packing A. If we won't be packing A, we alias to
				   the _pc variables so that code further down can unconditionally
				   reference the _pa variables. Note that *if* we will be packing
				   A, the thrinfo_t node will have already been created by a
				   previous call to bli_thrinfo_grow(), since bszid values of
				   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
				   bszid that is a normal bszid_t value). */ \
				bszid_t*   restrict bszids_pb; \
				if ( packb ) { bszids_pb = &bszids_ic[1]; \
							   thread_pb = bli_thrinfo_sub_node( thread_ic ); } \
				else         { bszids_pb = &bszids_ic[0]; \
							   thread_pb = thread_ic; } \
\
				/* Determine the packing buffer and related parameters for matrix
				   B. (If B will not be packed, then b_use will be set to point to
				   b and the _b_use strides will be set accordingly.) Then call
				   the packm sup variant chooser, which will call the appropriate
				   implementation based on the schema deduced from the stor_id.
				   NOTE: packing matrix B in this panel-block algorithm corresponds
				   to packing matrix A in the block-panel algorithm. */ \
				PASTEMAC(ch,packm_sup_b) \
				( \
				  packb, \
				  BLIS_BUFFER_FOR_A_BLOCK, /* This algorithm packs matrix B to */ \
				  stor_id,                 /* a "block of A".                  */ \
				  BLIS_NO_TRANSPOSE, \
				  KC,     MC,       /* This "block of A" is (at most) KC x MC. */ \
				  kc_cur, mc_cur, NR, \
				  &one_local, \
				  b_ic,   rs_b,      cs_b, \
				  &b_use, &rs_b_use, &cs_b_use, \
				                     &ps_b_use, \
				  cntx, \
				  rntm, \
				  &mem_b, \
				  thread_pb  \
				); \
\
				/* Alias b_use so that it's clear this is our current block of
				   matrix B. */ \
				ctype* restrict b_ic_use = b_use; \
\
				/* Embed the panel stride of B within the auxinfo_t object. The
				   millikernel will query and use this to iterate through
				   micropanels of B. */ \
				bli_auxinfo_set_ps_b( ps_b_use, &aux ); \
\
				/* Grow the thrinfo_t tree. */ \
				bszid_t*   restrict bszids_jr = &bszids_pb[1]; \
				                    thread_jr = bli_thrinfo_sub_node( thread_pb ); \
				bli_thrinfo_sup_grow( rntm, bszids_jr, thread_jr ); \
\
				/* Compute number of primary and leftover components of the JR loop. */ \
				dim_t jr_iter = ( nc_cur + MR - 1 ) / MR; \
				dim_t jr_left =   nc_cur % MR; \
\
				/* Compute the JR loop thread range for the current thread. */ \
				dim_t jr_start, jr_end; \
				bli_thread_range_sub( thread_jr, jr_iter, 1, FALSE, &jr_start, &jr_end ); \
\
				/* An optimization: allow the last jr iteration to contain up to MRE
				   rows of C and A. (If MRE > MR, the mkernel has agreed to handle
				   these cases.) Note that this prevents us from declaring jr_iter and
				   jr_left as const. NOTE: We forgo this optimization when packing A
				   since packing an extended edge case is not yet supported. */ \
				if ( !packa && !is_mt ) \
				if ( MRE != 0 && 1 < jr_iter && jr_left != 0 && jr_left <= MRE ) \
				{ \
					jr_iter--; jr_left += MR; \
				} \
\
				/* Loop over the m dimension (NR columns at a time). */ \
				/*for ( dim_t j = 0; j < jr_iter; j += 1 )*/ \
				for ( dim_t j = jr_start; j < jr_end; j += 1 ) \
				{ \
					const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? MR : jr_left ); \
\
					/*
					ctype* restrict a_jr = a_pc + j * jrstep_a; \
					*/ \
					ctype* restrict a_jr = a_pc_use + j * ps_a_use; \
					ctype* restrict c_jr = c_ic     + j * jrstep_c; \
\
					/*
					const dim_t ir_iter = ( mc_cur + NR - 1 ) / NR; \
					const dim_t ir_left =   mc_cur % NR; \
					*/ \
\
					/* Loop over the n dimension (MR rows at a time). */ \
					{ \
						/* Invoke the gemmsup millikernel. */ \
						gemmsup_ker \
						( \
						  conja, \
						  conjb, \
						  nr_cur, /* Notice: nr_cur <= MR. */ \
						  mc_cur, /* Recall: mc_cur partitions the n dimension! */ \
						  kc_cur, \
						  alpha_cast, \
						  a_jr,     rs_a_use, cs_a_use, \
						  b_ic_use, rs_b_use, cs_b_use, \
						  beta_use, \
						  c_jr,     rs_c,     cs_c, \
						  &aux, \
						  cntx  \
						); \
					} \
				} \
			} \
\
			/* NOTE: This barrier is only needed if we are packing A (since
			   that matrix is packed within the pc loop of this variant). */ \
			if ( packa ) bli_thread_barrier( thread_pa ); \
		} \
	} \
\
	/* Release any memory that was acquired for packing matrices A and B. */ \
	PASTEMAC(ch,packm_sup_finalize_mem_a) \
	( \
	  packa, \
	  rntm, \
	  &mem_a, \
	  thread_pa  \
	); \
	PASTEMAC(ch,packm_sup_finalize_mem_b) \
	( \
	  packb, \
	  rntm, \
	  &mem_b, \
	  thread_pb  \
	); \
\
/*
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: b1", kc_cur, nr_cur, b_jr, rs_b, cs_b, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: a1", mr_cur, kc_cur, a_ir, rs_a, cs_a, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%4.1f", "" ); \
*/ \
}

INSERT_GENTFUNC_L( gemmtsup, ref_var1n )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, uplo, varname ) \
\
void PASTEMACT(ch,opname,uplo,varname) \
     ( \
       bool             packa, \
       bool             packb, \
       conj_t           conja, \
       conj_t           conjb, \
       dim_t            m, \
       dim_t            n, \
       dim_t            k, \
       void*   restrict alpha, \
       void*   restrict a, inc_t rs_a, inc_t cs_a, \
       void*   restrict b, inc_t rs_b, inc_t cs_b, \
       void*   restrict beta, \
       void*   restrict c, inc_t rs_c, inc_t cs_c, \
       stor3_t          stor_id, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	/* If m or n is zero, return immediately. */ \
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* If k < 1 or alpha is zero, scale by beta and return. */ \
	if ( k < 1 || PASTEMAC(ch,eq0)( *(( ctype* )alpha) ) ) \
	{ \
		if ( bli_thread_am_ochief( thread ) ) \
		{ \
			PASTEMAC(ch,scalm) \
			( \
			  BLIS_NO_CONJUGATE, \
			  0, \
			  BLIS_NONUNIT_DIAG, \
			  BLIS_DENSE, \
			  m, n, \
			  beta, \
			  c, rs_c, cs_c \
			); \
		} \
		return; \
	} \
\
	/* This transposition of the stor3_t id value is inherent to variant 1.
	   The reason: we assume that variant 2 is the "main" variant. The
	   consequence of this is that we assume that the millikernels that
	   iterate over m are registered to the "primary" kernel group associated
	   with the kernel IO preference; similarly, mkernels that iterate over
	   n are assumed to be registered to the "non-primary" group associated
	   with the ("non-primary") anti-preference. Note that this pattern holds
	   regardless of whether the mkernel set has a row or column preference.)
	   See bli_l3_sup_int.c for a higher-level view of how this choice is made. */ \
	stor_id = bli_stor3_trans( stor_id ); \
\
	/* Query the context for various blocksizes. */ \
	dim_t NR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NR, cntx ); \
	dim_t MR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MR, cntx ); \
	dim_t NC0 = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NC, cntx ); \
	dim_t MC0 = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MC, cntx ); \
	dim_t KC0 = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_KC, cntx ); \
\
	/* Query the maximum blocksize for MR, which implies a maximum blocksize
	extension for the final iteration. */ \
	dim_t MRM = bli_cntx_get_l3_sup_tri_blksz_max_dt( dt, BLIS_MR, cntx ); \
	/* Query the context for the sup microkernel address and cast it to its
	   function pointer type. */ \
	PASTECH(ch,gemmsup_ker_ft) \
               gemmsup_ker = bli_cntx_get_l3_sup_tri_ker_dt( dt, stor_id, cntx ); \
\
	if( ( 0 == NR ) || ( 0 == MR ) || ( 0 == NC0 ) || ( 0 == MC0 ) || ( 0 == KC0 ) ) \
	{ \
		NR = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx ); \
		MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx ); \
		NC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx ); \
		MC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx ); \
		KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx ); \
		MRM = bli_cntx_get_l3_sup_blksz_max_dt( dt, BLIS_MR, cntx ); \
		gemmsup_ker = bli_cntx_get_l3_sup_ker_dt( dt, stor_id, cntx ); \
	} \
	const dim_t MRE = MRM - MR; \
\
	dim_t KC; \
	if      ( packa && packb ) \
	{ \
		KC = KC0; \
	} \
	else if ( packb ) \
	{ \
		if      ( stor_id == BLIS_RRR || \
				  stor_id == BLIS_CCC    ) KC = KC0; \
		else if ( stor_id == BLIS_RRC || \
				  stor_id == BLIS_CRC    ) KC = KC0; \
		else if ( stor_id == BLIS_RCR || \
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4; \
		else                               KC = KC0; \
	} \
	else if ( packa ) \
	{ \
		if      ( stor_id == BLIS_RRR || \
				  stor_id == BLIS_CCC    ) KC = (( KC0 / 2 ) / 2 ) * 2; \
		else if ( stor_id == BLIS_RRC || \
				  stor_id == BLIS_CRC    ) KC = KC0; \
		else if ( stor_id == BLIS_RCR || \
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4; \
		else                               KC = KC0; \
	} \
	else /* if ( !packa && !packb ) */ \
	{ \
		if      ( FALSE                  ) KC = KC0; \
		else if ( stor_id == BLIS_RRC || \
				  stor_id == BLIS_CRC    ) KC = KC0; \
		else if ( m <=   MR && n <=   NR ) KC = KC0; \
		else if ( m <= 2*MR && n <= 2*NR ) KC = KC0 / 2; \
		else if ( m <= 3*MR && n <= 3*NR ) KC = (( KC0 / 3 ) / 4 ) * 4; \
		else if ( m <= 4*MR && n <= 4*NR ) KC = KC0 / 4; \
		else                               KC = (( KC0 / 5 ) / 4 ) * 4; \
	} \
\
	/* Nudge NC up to a multiple of MR and MC up to a multiple of NR.
	   NOTE: This is unique to variant 1 (ie: not performed in variant 2)
	   because MC % MR == 0 and NC % NR == 0 is already enforced at runtime. */ \
	const dim_t NC  = bli_align_dim_to_mult( NC0, MR ); \
	const dim_t MC  = bli_align_dim_to_mult( MC0, NR ); \
\
	/* Compute partitioning step values for each matrix of each loop. */ \
	const inc_t jcstep_c = rs_c; \
	const inc_t jcstep_a = rs_a; \
\
	const inc_t pcstep_a = cs_a; \
	const inc_t pcstep_b = rs_b; \
\
	const inc_t icstep_c = cs_c; \
	const inc_t icstep_b = cs_b; \
\
	const inc_t jrstep_c = rs_c * MR; \
\
	/*
	const inc_t jrstep_a = rs_a * MR; \
\
	const inc_t irstep_c = cs_c * NR; \
	const inc_t irstep_b = cs_b * NR; \
	*/ \
\
	ctype* restrict a_00       = a; \
	ctype* restrict b_00       = b; \
	ctype* restrict c_00       = c; \
	ctype* restrict alpha_cast = alpha; \
	ctype* restrict beta_cast  = beta; \
\
	/* Make local copies of beta and one scalars to prevent any unnecessary
	   sharing of cache lines between the cores' caches. */ \
	ctype           beta_local = *beta_cast; \
	ctype           one_local  = *PASTEMAC(ch,1); \
\
	auxinfo_t       aux; \
\
	/* Parse and interpret the contents of the rntm_t object to properly
	   set the ways of parallelism for each loop. */ \
	/*bli_rntm_set_ways_from_rntm_sup( m, n, k, rntm );*/ \
\
	/* Initialize a mem_t entry for A and B. Strictly speaking, this is only
	   needed for the matrix we will be packing (if any), but we do it
	   unconditionally to be safe. An alternative way of initializing the
	   mem_t entries is:

	     bli_mem_clear( &mem_a ); \
	     bli_mem_clear( &mem_b ); \
	*/ \
	mem_t mem_a = BLIS_MEM_INITIALIZER; \
	mem_t mem_b = BLIS_MEM_INITIALIZER; \
\
	/* Define an array of bszid_t ids, which will act as our substitute for
	   the cntl_t tree.
	   NOTE: These bszid_t values, and their order, match that of the bp
	   algorithm (variant 2) because they are not used to query actual
	   blocksizes but rather query the ways of parallelism for the various
	   loops. For example, the 2nd loop in variant 1 partitions in the m
	   dimension (in increments of MR), but parallelizes that m dimension
	   with BLIS_JR_NT. The only difference is that the _packa and _packb
	   arrays have been adjusted for the semantic difference in order in
	   which packa and packb nodes are encountered in the thrinfo tree.
	   That is, this panel-block algorithm partitions an NC x KC submatrix
	   of A to be packed in the 4th loop, and a KC x MC submatrix of B
	   to be packed in the 3rd loop. */ \
	/*                           5thloop  4thloop         packa  3rdloop         packb  2ndloop  1stloop  ukrloop */ \
	bszid_t bszids_nopack[6] = { BLIS_NC, BLIS_KC,               BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t bszids_packa [7] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t bszids_packb [7] = { BLIS_NC, BLIS_KC,               BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t bszids_packab[8] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t* restrict bszids; \
\
	/* Set the bszids pointer to the correct bszids array above based on which
	   matrices (if any) are being packed. */ \
	if ( packa ) { if ( packb ) bszids = bszids_packab; \
	               else         bszids = bszids_packa; } \
	else         { if ( packb ) bszids = bszids_packb; \
	               else         bszids = bszids_nopack; } \
\
	/* Determine whether we are using more than one thread. */ \
	const bool is_mt = bli_rntm_calc_num_threads( rntm ); \
\
	thrinfo_t* restrict thread_jc = NULL; \
	thrinfo_t* restrict thread_pc = NULL; \
	thrinfo_t* restrict thread_pa = NULL; \
	thrinfo_t* restrict thread_ic = NULL; \
	thrinfo_t* restrict thread_pb = NULL; \
	thrinfo_t* restrict thread_jr = NULL; \
\
	/* Grow the thrinfo_t tree. */ \
	bszid_t*   restrict bszids_jc = bszids; \
	                    thread_jc = thread; \
	bli_thrinfo_sup_grow( rntm, bszids_jc, thread_jc ); \
\
	/* Compute the JC loop thread range for the current thread. */ \
	dim_t jc_start, jc_end; \
	bli_thread_range_sub( thread_jc, m, MR, FALSE, &jc_start, &jc_end ); \
	const dim_t m_local = jc_end - jc_start; \
\
	/* Compute number of primary and leftover components of the JC loop. */ \
	/*const dim_t jc_iter = ( m_local + NC - 1 ) / NC;*/ \
	const dim_t jc_left =   m_local % NC; \
\
	/* Loop over the m dimension (NC rows/columns at a time). */ \
	/*for ( dim_t jj = 0; jj < jc_iter; jj += 1 )*/ \
	for ( dim_t jj = jc_start; jj < jc_end; jj += NC ) \
	{ \
		/* Calculate the thread's current JC block dimension. */ \
		const dim_t nc_cur = ( NC <= jc_end - jj ? NC : jc_left ); \
\
		ctype* restrict a_jc = a_00 + jj * jcstep_a; \
		ctype* restrict c_jc = c_00 + jj * jcstep_c; \
\
		/* Grow the thrinfo_t tree. */ \
		bszid_t*   restrict bszids_pc = &bszids_jc[1]; \
		                    thread_pc = bli_thrinfo_sub_node( thread_jc ); \
		bli_thrinfo_sup_grow( rntm, bszids_pc, thread_pc ); \
\
		/* Compute the PC loop thread range for the current thread. */ \
		const dim_t pc_start = 0, pc_end = k; \
		const dim_t k_local = k; \
\
		/* Compute number of primary and leftover components of the PC loop. */ \
		/*const dim_t pc_iter = ( k_local + KC - 1 ) / KC;*/ \
		const dim_t pc_left =   k_local % KC; \
\
		/* Loop over the k dimension (KC rows/columns at a time). */ \
		/*for ( dim_t pp = 0; pp < pc_iter; pp += 1 )*/ \
		for ( dim_t pp = pc_start; pp < pc_end; pp += KC ) \
		{ \
			/* Calculate the thread's current PC block dimension. */ \
			const dim_t kc_cur = ( KC <= pc_end - pp ? KC : pc_left ); \
\
			ctype* restrict a_pc = a_jc + pp * pcstep_a; \
			ctype* restrict b_pc = b_00 + pp * pcstep_b; \
\
			/* Only apply beta to the first iteration of the pc loop. */ \
			ctype* restrict beta_use = ( pp == 0 ? &beta_local : &one_local ); \
\
			ctype* a_use; \
			inc_t  rs_a_use, cs_a_use, ps_a_use; \
\
			/* Set the bszid_t array and thrinfo_t pointer based on whether
			   we will be packing A. If we won't be packing A, we alias to
			   the _pc variables so that code further down can unconditionally
			   reference the _pa variables. Note that *if* we will be packing
			   A, the thrinfo_t node will have already been created by a
			   previous call to bli_thrinfo_grow(), since bszid values of
			   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
			   bszid that is a normal bszid_t value). */ \
			bszid_t*   restrict bszids_pa; \
			if ( packa ) { bszids_pa = &bszids_pc[1]; \
			               thread_pa = bli_thrinfo_sub_node( thread_pc ); } \
			else         { bszids_pa = &bszids_pc[0]; \
			               thread_pa = thread_pc; } \
\
			/* Determine the packing buffer and related parameters for matrix
			   A. (If A will not be packed, then a_use will be set to point to
			   a and the _a_use strides will be set accordingly.) Then call
			   the packm sup variant chooser, which will call the appropriate
			   implementation based on the schema deduced from the stor_id.
			   NOTE: packing matrix A in this panel-block algorithm corresponds
			   to packing matrix B in the block-panel algorithm. */ \
			PASTEMAC(ch,packm_sup_a) \
			( \
			  packa, \
			  BLIS_BUFFER_FOR_B_PANEL, /* This algorithm packs matrix A to */ \
			  stor_id,                 /* a "panel of B".                  */ \
			  BLIS_NO_TRANSPOSE, \
			  NC,     KC,       /* This "panel of B" is (at most) NC x KC. */ \
			  nc_cur, kc_cur, MR, \
			  &one_local, \
			  a_pc,   rs_a,      cs_a, \
			  &a_use, &rs_a_use, &cs_a_use, \
			                     &ps_a_use, \
			  cntx, \
			  rntm, \
			  &mem_a, \
			  thread_pa  \
			); \
\
			/* Alias a_use so that it's clear this is our current block of
			   matrix A. */ \
			ctype* restrict a_pc_use = a_use; \
\
			/* We don't need to embed the panel stride of A within the auxinfo_t
			   object because this variant iterates through A in the jr loop,
			   which occurs here, within the macrokernel, not within the
			   millikernel. */ \
			/*bli_auxinfo_set_ps_a( ps_a_use, &aux );*/ \
\
			/* Grow the thrinfo_t tree. */ \
			bszid_t*   restrict bszids_ic = &bszids_pa[1]; \
			                    thread_ic = bli_thrinfo_sub_node( thread_pa ); \
			bli_thrinfo_sup_grow( rntm, bszids_ic, thread_ic ); \
\
			/* Compute the IC loop thread range for the current thread. */ \
			dim_t ic_start, ic_end; \
			bli_thread_range_sub( thread_ic, n, NR, FALSE, &ic_start, &ic_end ); \
			const dim_t n_local = ic_end - ic_start; \
\
			/* Compute number of primary and leftover components of the IC loop. */ \
			/*const dim_t ic_iter = ( n_local + MC - 1 ) / MC;*/ \
			const dim_t ic_left =   n_local % MC; \
\
			/* Loop over the n dimension (MC rows at a time). */ \
			/*for ( dim_t ii = 0; ii < ic_iter; ii += 1 )*/ \
			for ( dim_t ii = ic_start; ii < ic_end; ii += MC ) \
			{ \
				/* Calculate the thread's current IC block dimension. */ \
				const dim_t mc_cur = ( MC <= ic_end - ii ? MC : ic_left ); \
\
				ctype* restrict b_ic = b_pc + ii * icstep_b; \
				ctype* restrict c_ic = c_jc + ii * icstep_c; \
\
				ctype* b_use; \
				inc_t  rs_b_use, cs_b_use, ps_b_use; \
\
				/* Set the bszid_t array and thrinfo_t pointer based on whether
				   we will be packing A. If we won't be packing A, we alias to
				   the _pc variables so that code further down can unconditionally
				   reference the _pa variables. Note that *if* we will be packing
				   A, the thrinfo_t node will have already been created by a
				   previous call to bli_thrinfo_grow(), since bszid values of
				   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
				   bszid that is a normal bszid_t value). */ \
				bszid_t*   restrict bszids_pb; \
				if ( packb ) { bszids_pb = &bszids_ic[1]; \
							   thread_pb = bli_thrinfo_sub_node( thread_ic ); } \
				else         { bszids_pb = &bszids_ic[0]; \
							   thread_pb = thread_ic; } \
\
				/* Determine the packing buffer and related parameters for matrix
				   B. (If B will not be packed, then b_use will be set to point to
				   b and the _b_use strides will be set accordingly.) Then call
				   the packm sup variant chooser, which will call the appropriate
				   implementation based on the schema deduced from the stor_id.
				   NOTE: packing matrix B in this panel-block algorithm corresponds
				   to packing matrix A in the block-panel algorithm. */ \
				PASTEMAC(ch,packm_sup_b) \
				( \
				  packb, \
				  BLIS_BUFFER_FOR_A_BLOCK, /* This algorithm packs matrix B to */ \
				  stor_id,                 /* a "block of A".                  */ \
				  BLIS_NO_TRANSPOSE, \
				  KC,     MC,       /* This "block of A" is (at most) KC x MC. */ \
				  kc_cur, mc_cur, NR, \
				  &one_local, \
				  b_ic,   rs_b,      cs_b, \
				  &b_use, &rs_b_use, &cs_b_use, \
				                     &ps_b_use, \
				  cntx, \
				  rntm, \
				  &mem_b, \
				  thread_pb  \
				); \
\
				/* Alias b_use so that it's clear this is our current block of
				   matrix B. */ \
				ctype* restrict b_ic_use = b_use; \
\
				/* Embed the panel stride of B within the auxinfo_t object. The
				   millikernel will query and use this to iterate through
				   micropanels of B. */ \
				bli_auxinfo_set_ps_b( ps_b_use, &aux ); \
\
				/* Grow the thrinfo_t tree. */ \
				bszid_t*   restrict bszids_jr = &bszids_pb[1]; \
				                    thread_jr = bli_thrinfo_sub_node( thread_pb ); \
				bli_thrinfo_sup_grow( rntm, bszids_jr, thread_jr ); \
\
				/* Compute number of primary and leftover components of the JR loop. */ \
				dim_t jr_iter = ( nc_cur + MR - 1 ) / MR; \
				dim_t jr_left =   nc_cur % MR; \
\
				/* Compute the JR loop thread range for the current thread. */ \
				dim_t jr_start, jr_end; \
				bli_thread_range_sub( thread_jr, jr_iter, 1, FALSE, &jr_start, &jr_end ); \
\
				/* An optimization: allow the last jr iteration to contain up to MRE
				   rows of C and A. (If MRE > MR, the mkernel has agreed to handle
				   these cases.) Note that this prevents us from declaring jr_iter and
				   jr_left as const. NOTE: We forgo this optimization when packing A
				   since packing an extended edge case is not yet supported. */ \
				if ( !packa && !is_mt ) \
				if ( MRE != 0 && 1 < jr_iter && jr_left != 0 && jr_left <= MRE ) \
				{ \
					jr_iter--; jr_left += MR; \
				} \
\
				/* Loop over the m dimension (NR columns at a time). */ \
				/*for ( dim_t j = 0; j < jr_iter; j += 1 )*/ \
				for ( dim_t j = jr_start; j < jr_end; j += 1 ) \
				{ \
					const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? MR : jr_left ); \
\
					/*
					ctype* restrict a_jr = a_pc + j * jrstep_a; \
					*/ \
					ctype* restrict a_jr = a_pc_use + j * ps_a_use; \
					ctype* restrict c_jr = c_ic     + j * jrstep_c; \
\
					/*
					const dim_t ir_iter = ( mc_cur + NR - 1 ) / NR; \
					const dim_t ir_left =   mc_cur % NR; \
					*/ \
\
					/* Loop over the n dimension (MR rows at a time). */ \
					{ \
						/* Invoke the gemmsup millikernel. */ \
						gemmsup_ker \
						( \
						  conja, \
						  conjb, \
						  nr_cur, /* Notice: nr_cur <= MR. */ \
						  mc_cur, /* Recall: mc_cur partitions the n dimension! */ \
						  kc_cur, \
						  alpha_cast, \
						  a_jr,     rs_a_use, cs_a_use, \
						  b_ic_use, rs_b_use, cs_b_use, \
						  beta_use, \
						  c_jr,     rs_c,     cs_c, \
						  &aux, \
						  cntx  \
						); \
					} \
				} \
			} \
\
			/* NOTE: This barrier is only needed if we are packing A (since
			   that matrix is packed within the pc loop of this variant). */ \
			if ( packa ) bli_thread_barrier( thread_pa ); \
		} \
	} \
\
	/* Release any memory that was acquired for packing matrices A and B. */ \
	PASTEMAC(ch,packm_sup_finalize_mem_a) \
	( \
	  packa, \
	  rntm, \
	  &mem_a, \
	  thread_pa  \
	); \
	PASTEMAC(ch,packm_sup_finalize_mem_b) \
	( \
	  packb, \
	  rntm, \
	  &mem_b, \
	  thread_pb  \
	); \
\
/*
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: b1", kc_cur, nr_cur, b_jr, rs_b, cs_b, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: a1", mr_cur, kc_cur, a_ir, rs_a, cs_a, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%4.1f", "" ); \
*/ \
}

INSERT_GENTFUNC_U( gemmtsup, ref_var1n )


//
// -- var2m --------------------------------------------------------------------
//

static FUNCPTR_T GENARRAY_T(ftypes_var2m,gemmtsup,ref_var2m);

void bli_gemmtsup_ref_var2m
     (
       trans_t trans,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       stor3_t eff_id,
       cntx_t* cntx,
       rntm_t* rntm,
       thrinfo_t* thread
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_5);
#if 0
	obj_t at, bt;

	bli_obj_alias_to( a, &at );
	bli_obj_alias_to( b, &bt );

	// Induce transpositions on A and/or B if either object is marked for
	// transposition. We can induce "fast" transpositions since they objects
	// are guaranteed to not have structure or be packed.
	if ( bli_obj_has_trans( &at ) ) { bli_obj_induce_fast_trans( &at ); }
	if ( bli_obj_has_trans( &bt ) ) { bli_obj_induce_fast_trans( &bt ); }

	const num_t    dt        = bli_obj_dt( c );

	const conj_t   conja     = bli_obj_conj_status( a );
	const conj_t   conjb     = bli_obj_conj_status( b );

	const dim_t    m         = bli_obj_length( c );
	const dim_t    n         = bli_obj_width( c );

	const dim_t    k         = bli_obj_width( &at );

	void* restrict buf_a     = bli_obj_buffer_at_off( &at );
	const inc_t    rs_a      = bli_obj_row_stride( &at );
	const inc_t    cs_a      = bli_obj_col_stride( &at );

	void* restrict buf_b     = bli_obj_buffer_at_off( &bt );
	const inc_t    rs_b      = bli_obj_row_stride( &bt );
	const inc_t    cs_b      = bli_obj_col_stride( &bt );

	void* restrict buf_c     = bli_obj_buffer_at_off( c );
	const inc_t    rs_c      = bli_obj_row_stride( c );
	const inc_t    cs_c      = bli_obj_col_stride( c );

	void* restrict buf_alpha = bli_obj_buffer_for_1x1( dt, alpha );
	void* restrict buf_beta  = bli_obj_buffer_for_1x1( dt, beta );

#else
	const num_t    dt        = bli_obj_dt( c );

	const bool     packa     = bli_rntm_pack_a( rntm );
	const bool     packb     = bli_rntm_pack_b( rntm );

	const conj_t   conja     = bli_obj_conj_status( a );
	const conj_t   conjb     = bli_obj_conj_status( b );

	const dim_t    m         = bli_obj_length( c );
	const dim_t    n         = bli_obj_width( c );
	      dim_t    k;

	void* restrict buf_a = bli_obj_buffer_at_off( a );
	      inc_t    rs_a;
	      inc_t    cs_a;

	void* restrict buf_b = bli_obj_buffer_at_off( b );
	      inc_t    rs_b;
	      inc_t    cs_b;

	if ( bli_obj_has_notrans( a ) )
	{
		k     = bli_obj_width( a );

		rs_a  = bli_obj_row_stride( a );
		cs_a  = bli_obj_col_stride( a );
	}
	else // if ( bli_obj_has_trans( a ) )
	{
		// Assign the variables with an implicit transposition.
		k     = bli_obj_length( a );

		rs_a  = bli_obj_col_stride( a );
		cs_a  = bli_obj_row_stride( a );
	}

	if ( bli_obj_has_notrans( b ) )
	{
		rs_b  = bli_obj_row_stride( b );
		cs_b  = bli_obj_col_stride( b );
	}
	else // if ( bli_obj_has_trans( b ) )
	{
		// Assign the variables with an implicit transposition.
		rs_b  = bli_obj_col_stride( b );
		cs_b  = bli_obj_row_stride( b );
	}


	// Optimize some storage/packing cases by transforming them into others.
	// These optimizations are expressed by changing trans and/or eff_id.
	bli_gemmsup_ref_var1n2m_opt_cases( dt, &trans, packa, packb, &eff_id, cntx );


	bool uploc;
	if ( bli_is_notrans ( trans ) )
		uploc = bli_obj_is_lower( c ) ? 0 : 1;
	else
		uploc = bli_obj_is_lower( c ) ? 1 : 0;

	void* restrict buf_c     = bli_obj_buffer_at_off( c );
	const inc_t    rs_c      = bli_obj_row_stride( c );
	const inc_t    cs_c      = bli_obj_col_stride( c );

	void* restrict buf_alpha = bli_obj_buffer_for_1x1( dt, alpha );
	void* restrict buf_beta  = bli_obj_buffer_for_1x1( dt, beta );

#endif

	// Index into the type combination array to extract the correct
	// function pointer.
	FUNCPTR_T f = ftypes_var2m[dt][uploc];



	if ( bli_is_notrans( trans ) )
	{
		// Invoke the function.
		f
		(
		  packa,
		  packb,
		  conja,
		  conjb,
		  m,
		  n,
		  k,
		  buf_alpha,
		  buf_a, rs_a, cs_a,
		  buf_b, rs_b, cs_b,
		  buf_beta,
		  buf_c, rs_c, cs_c,
		  eff_id,
		  cntx,
		  rntm,
		  thread
		);
	}
	else
	{
		// Invoke the function (transposing the operation).
		f
		(
		  packb,             // swap the pack values.
		  packa,
		  conjb,             // swap the conj values.
		  conja,
		  n,                 // swap the m and n dimensions.
		  m,
		  k,
		  buf_alpha,
		  buf_b, cs_b, rs_b, // swap the positions of A and B.
		  buf_a, cs_a, rs_a, // swap the strides of A and B.
		  buf_beta,
		  buf_c, cs_c, rs_c, // swap the strides of C.
		  bli_stor3_trans( eff_id ), // transpose the stor3_t id.
		  cntx,
		  rntm,
		  thread
		);
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_5);
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, uplo, varname ) \
\
void PASTEMACT(ch,opname,uplo,varname) \
     ( \
       bool             packa, \
       bool             packb, \
       conj_t           conja, \
       conj_t           conjb, \
       dim_t            m, \
       dim_t            n, \
       dim_t            k, \
       void*   restrict alpha, \
       void*   restrict a, inc_t rs_a, inc_t cs_a, \
       void*   restrict b, inc_t rs_b, inc_t cs_b, \
       void*   restrict beta, \
       void*   restrict c, inc_t rs_c, inc_t cs_c, \
       stor3_t          stor_id, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype* restrict zero = PASTEMAC(ch,0); \
\
	/* If m or n is zero, return immediately. */ \
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* If k < 1 or alpha is zero, scale by beta and return. */ \
	if ( k < 1 || PASTEMAC(ch,eq0)( *(( ctype* )alpha) ) ) \
	{ \
		if ( bli_thread_am_ochief( thread ) ) \
		{ \
			PASTEMAC(ch,scalm) \
			( \
			  BLIS_NO_CONJUGATE, \
			  0, \
			  BLIS_NONUNIT_DIAG, \
			  BLIS_DENSE, \
			  m, n, \
			  beta, \
			  c, rs_c, cs_c \
			); \
		} \
		return; \
	} \
\
	/* Query the context for various blocksizes. */ \
	dim_t NR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NR, cntx ); \
	dim_t MR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MR, cntx ); \
	dim_t NC  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NC, cntx ); \
	dim_t MC  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MC, cntx ); \
	dim_t KC0 = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_KC, cntx ); \
	/* Query the maximum blocksize for NR, which implies a maximum blocksize
	   extension for the final iteration. */ \
	dim_t NRM = bli_cntx_get_l3_sup_tri_blksz_max_dt( dt, BLIS_NR, cntx ); \
\
	/* Query the context for the sup microkernel address and cast it to its
	   function pointer type. */ \
	PASTECH(ch,gemmsup_ker_ft) \
               gemmsup_ker = bli_cntx_get_l3_sup_tri_ker_dt( dt, stor_id, cntx ); \
\
	if( ( 0 == NR ) || ( 0 == MR ) || ( 0 == NC ) || ( 0 == MC ) || ( 0 == KC0 ) ) \
	{ \
		NR = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx ); \
		MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx ); \
		NC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx ); \
		MC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx ); \
		KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx ); \
		NRM = bli_cntx_get_l3_sup_blksz_max_dt( dt, BLIS_NR, cntx ); \
		gemmsup_ker = bli_cntx_get_l3_sup_ker_dt( dt, stor_id, cntx ); \
	} \
	const dim_t NRE = NRM - NR; \
\
	dim_t KC; \
	if      ( packa && packb ) \
	{ \
		KC = KC0; \
	} \
	else if ( packb ) \
	{ \
		if      ( stor_id == BLIS_RRR || \
				  stor_id == BLIS_CCC    ) KC = KC0; \
		else if ( stor_id == BLIS_RRC || \
				  stor_id == BLIS_CRC    ) KC = KC0; \
		else if ( stor_id == BLIS_RCR || \
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4; \
		else                               KC = KC0; \
	} \
	else if ( packa ) \
	{ \
		if      ( stor_id == BLIS_RRR || \
				  stor_id == BLIS_CCC    ) KC = (( KC0 / 2 ) / 2 ) * 2; \
		else if ( stor_id == BLIS_RRC || \
				  stor_id == BLIS_CRC    ) KC = KC0; \
		else if ( stor_id == BLIS_RCR || \
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4; \
		else                               KC = KC0; \
	} \
	else /* if ( !packa && !packb ) */ \
	{ \
		if      ( stor_id == BLIS_RRR || \
				  stor_id == BLIS_CCC    ) KC = KC0; \
		else if ( stor_id == BLIS_RRC || \
				  stor_id == BLIS_CRC    ) KC = KC0; \
		else if ( m <=   MR && n <=   NR ) KC = KC0; \
		else if ( m <= 2*MR && n <= 2*NR ) KC = KC0 / 2; \
		else if ( m <= 3*MR && n <= 3*NR ) KC = (( KC0 / 3 ) / 4 ) * 4; \
		else if ( m <= 4*MR && n <= 4*NR ) KC = KC0 / 4; \
		else                               KC = (( KC0 / 5 ) / 4 ) * 4; \
	} \
\
	/* Compute partitioning step values for each matrix of each loop. */ \
	const inc_t jcstep_c = cs_c; \
	const inc_t jcstep_b = cs_b; \
\
	const inc_t pcstep_a = cs_a; \
	const inc_t pcstep_b = rs_b; \
\
	const inc_t icstep_c = rs_c; \
	const inc_t icstep_a = rs_a; \
\
	const inc_t jrstep_c = cs_c * NR; \
\
	const inc_t irstep_c = rs_c * MR; \
\
	/*
	const inc_t jrstep_b = cs_b * NR; \
	( void )jrstep_b; \
\
	const inc_t irstep_c = rs_c * MR; \
	const inc_t irstep_a = rs_a * MR; \
	*/ \
\
	ctype ct[ BLIS_STACK_BUF_MAX_SIZE / sizeof( ctype ) ]  __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
\
	/* storage-scheme of ct should be same as that of C.
	  Since update routines only support row-major order,
	  col_pref flag is used to induce transpose to matrices before
	  passing to update routine whenever C is col-stored */ \
	const bool col_pref = (rs_c == 1)? 1 : 0; \
\
	const inc_t rs_ct = ( col_pref ? 1 : NR ); \
	const inc_t cs_ct = ( col_pref ? MR : 1 ); \
\
	ctype* restrict a_00       = a; \
	ctype* restrict b_00       = b; \
	ctype* restrict c_00       = c; \
	ctype* restrict alpha_cast = alpha; \
	ctype* restrict beta_cast  = beta; \
\
	/* Make local copies of beta and one scalars to prevent any unnecessary
	   sharing of cache lines between the cores' caches. */ \
	ctype           beta_local = *beta_cast; \
	ctype           one_local  = *PASTEMAC(ch,1); \
\
	auxinfo_t       aux; \
\
	/* Parse and interpret the contents of the rntm_t object to properly
	   set the ways of parallelism for each loop. */ \
	/*bli_rntm_set_ways_from_rntm_sup( m, n, k, rntm );*/ \
\
	/* Initialize a mem_t entry for A and B. Strictly speaking, this is only
	   needed for the matrix we will be packing (if any), but we do it
	   unconditionally to be safe. An alternative way of initializing the
	   mem_t entries is:

	     bli_mem_clear( &mem_a ); \
	     bli_mem_clear( &mem_b ); \
	*/ \
	mem_t mem_a = BLIS_MEM_INITIALIZER; \
	mem_t mem_b = BLIS_MEM_INITIALIZER; \
\
	/* Define an array of bszid_t ids, which will act as our substitute for
	   the cntl_t tree. */ \
	/*                           5thloop  4thloop         packb  3rdloop         packa  2ndloop  1stloop  ukrloop */ \
	bszid_t bszids_nopack[6] = { BLIS_NC, BLIS_KC,               BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t bszids_packa [7] = { BLIS_NC, BLIS_KC,               BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t bszids_packb [7] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t bszids_packab[8] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t* restrict bszids; \
\
	/* Set the bszids pointer to the correct bszids array above based on which
	   matrices (if any) are being packed. */ \
	if ( packa ) { if ( packb ) bszids = bszids_packab; \
	               else         bszids = bszids_packa; } \
	else         { if ( packb ) bszids = bszids_packb; \
	               else         bszids = bszids_nopack; } \
\
	/* Determine whether we are using more than one thread. */ \
	const bool is_mt = bli_rntm_calc_num_threads( rntm ); \
\
	thrinfo_t* restrict thread_jc = NULL; \
	thrinfo_t* restrict thread_pc = NULL; \
	thrinfo_t* restrict thread_pb = NULL; \
	thrinfo_t* restrict thread_ic = NULL; \
	thrinfo_t* restrict thread_pa = NULL; \
	thrinfo_t* restrict thread_jr = NULL; \
\
	/* Grow the thrinfo_t tree. */ \
	bszid_t*   restrict bszids_jc = bszids; \
	                    thread_jc = thread; \
	bli_thrinfo_sup_grow( rntm, bszids_jc, thread_jc ); \
\
	/* Compute the JC loop thread range for the current thread. */ \
	dim_t jc_start, jc_end; \
	bli_thread_range_weighted_sub( thread_jc, 0, BLIS_LOWER, m, n, NR, FALSE, &jc_start, &jc_end ); \
	const dim_t n_local = jc_end - jc_start; \
\
	/* Compute number of primary and leftover components of the JC loop. */ \
	/*const dim_t jc_iter = ( n_local + NC - 1 ) / NC;*/ \
	const dim_t jc_left =   n_local % NC; \
\
	dim_t m_off_cblock, n_off_cblock; \
	dim_t m_off = 0; \
	dim_t n_off = 0; \
	doff_t diagoffc; \
	dim_t i, ip; \
\
	/* Loop over the n dimension (NC rows/columns at a time). */ \
	/*for ( dim_t jj = 0; jj < jc_iter; jj += 1 )*/ \
	for ( dim_t jj = jc_start; jj < jc_end; jj += NC ) \
	{ \
		/* Calculate the thread's current JC block dimension. */ \
		const dim_t nc_cur = ( NC <= jc_end - jj ? NC : jc_left ); \
\
		ctype* restrict b_jc = b_00 + jj * jcstep_b; \
		ctype* restrict c_jc = c_00 + jj * jcstep_c; \
\
		/* Grow the thrinfo_t tree. */ \
		bszid_t*   restrict bszids_pc = &bszids_jc[1]; \
		                    thread_pc = bli_thrinfo_sub_node( thread_jc ); \
		bli_thrinfo_sup_grow( rntm, bszids_pc, thread_pc ); \
\
		/* Compute the PC loop thread range for the current thread. */ \
		const dim_t pc_start = 0, pc_end = k; \
		const dim_t k_local = k; \
\
		/* Compute number of primary and leftover components of the PC loop. */ \
		/*const dim_t pc_iter = ( k_local + KC - 1 ) / KC;*/ \
		const dim_t pc_left =   k_local % KC; \
\
		/* Loop over the k dimension (KC rows/columns at a time). */ \
		/*for ( dim_t pp = 0; pp < pc_iter; pp += 1 )*/ \
		for ( dim_t pp = pc_start; pp < pc_end; pp += KC ) \
		{ \
			/* Calculate the thread's current PC block dimension. */ \
			const dim_t kc_cur = ( KC <= pc_end - pp ? KC : pc_left ); \
\
			ctype* restrict a_pc = a_00 + pp * pcstep_a; \
			ctype* restrict b_pc = b_jc + pp * pcstep_b; \
\
			/* Only apply beta to the first iteration of the pc loop. */ \
			ctype* restrict beta_use = ( pp == 0 ? &beta_local : &one_local ); \
\
			m_off = 0; \
			n_off = jj; \
			diagoffc = m_off - n_off; \
\
			ctype* b_use; \
			inc_t  rs_b_use, cs_b_use, ps_b_use; \
\
			/* Set the bszid_t array and thrinfo_t pointer based on whether
			   we will be packing B. If we won't be packing B, we alias to
			   the _pc variables so that code further down can unconditionally
			   reference the _pb variables. Note that *if* we will be packing
			   B, the thrinfo_t node will have already been created by a
			   previous call to bli_thrinfo_grow(), since bszid values of
			   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
			   bszid that is a normal bszid_t value). */ \
			bszid_t*   restrict bszids_pb; \
			if ( packb ) { bszids_pb = &bszids_pc[1]; \
			               thread_pb = bli_thrinfo_sub_node( thread_pc ); } \
			else         { bszids_pb = &bszids_pc[0]; \
			               thread_pb = thread_pc; } \
\
			/* Determine the packing buffer and related parameters for matrix
			   B. (If B will not be packed, then a_use will be set to point to
			   b and the _b_use strides will be set accordingly.) Then call
			   the packm sup variant chooser, which will call the appropriate
			   implementation based on the schema deduced from the stor_id. */ \
			PASTEMAC(ch,packm_sup_b) \
			( \
			  packb, \
			  BLIS_BUFFER_FOR_B_PANEL, /* This algorithm packs matrix B to */ \
			  stor_id,                 /* a "panel of B."                  */ \
			  BLIS_NO_TRANSPOSE, \
			  KC,     NC,       /* This "panel of B" is (at most) KC x NC. */ \
			  kc_cur, nc_cur, NR, \
			  &one_local, \
			  b_pc,   rs_b,      cs_b, \
			  &b_use, &rs_b_use, &cs_b_use, \
			                     &ps_b_use, \
			  cntx, \
			  rntm, \
			  &mem_b, \
			  thread_pb  \
			); \
\
			/* Alias a_use so that it's clear this is our current block of
			   matrix B. */ \
			ctype* restrict b_pc_use = b_use; \
\
			/* We don't need to embed the panel stride of B within the auxinfo_t
			   object because this variant iterates through B in the jr loop,
			   which occurs here, within the macrokernel, not within the
			   millikernel. */ \
			/*bli_auxinfo_set_ps_b( ps_b_use, &aux );*/ \
\
			/* Grow the thrinfo_t tree. */ \
			bszid_t*   restrict bszids_ic = &bszids_pb[1]; \
			                    thread_ic = bli_thrinfo_sub_node( thread_pb ); \
			bli_thrinfo_sup_grow( rntm, bszids_ic, thread_ic ); \
\
			/* Compute the IC loop thread range for the current thread. */ \
			dim_t ic_start, ic_end; \
			bli_thread_range_weighted_sub( thread_ic, -diagoffc, BLIS_UPPER, nc_cur, m, MR, FALSE, &ic_start, &ic_end ); \
			const dim_t m_local = ic_end - ic_start; \
\
			/* Compute number of primary and leftover components of the IC loop. */ \
			/*const dim_t ic_iter = ( m_local + MC - 1 ) / MC;*/ \
			const dim_t ic_left =   m_local % MC; \
\
			/* Loop over the m dimension (MC rows at a time). */ \
			/*for ( dim_t ii = 0; ii < ic_iter; ii += 1 )*/ \
			for ( dim_t ii = ic_start; ii < ic_end; ii += MC ) \
			{ \
				/* Calculate the thread's current IC block dimension. */ \
				dim_t mc_cur = ( MC <= ic_end - ii ? MC : ic_left ); \
				dim_t nc_pruned = nc_cur; \
\
				ctype* restrict a_ic = a_pc + ii * icstep_a; \
				ctype* restrict c_ic = c_jc + ii * icstep_c; \
\
				m_off = ii; \
\
				if(bli_gemmt_is_strictly_above_diag( m_off, n_off, mc_cur, nc_cur ) ) continue; \
\
				diagoffc = m_off - n_off; \
\
				if( diagoffc < 0 ) \
				{ \
					ip = -diagoffc / MR; \
					i = ip * MR; \
					mc_cur = mc_cur - i; \
					diagoffc = -diagoffc % MR; \
					m_off += i; \
					c_ic = c_ic + ( i ) * rs_c; \
					a_ic = a_ic + ( i ) * rs_a; \
				} \
\
				if( ( diagoffc + mc_cur ) < nc_cur ) \
				{ \
					nc_pruned = diagoffc + mc_cur; \
				} \
\
				ctype* a_use; \
				inc_t  rs_a_use, cs_a_use, ps_a_use; \
\
				/* Set the bszid_t array and thrinfo_t pointer based on whether
				   we will be packing B. If we won't be packing A, we alias to
				   the _ic variables so that code further down can unconditionally
				   reference the _pa variables. Note that *if* we will be packing
				   A, the thrinfo_t node will have already been created by a
				   previous call to bli_thrinfo_grow(), since bszid values of
				   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
				   bszid that is a normal bszid_t value). */ \
				bszid_t*   restrict bszids_pa; \
				if ( packa ) { bszids_pa = &bszids_ic[1]; \
							   thread_pa = bli_thrinfo_sub_node( thread_ic ); } \
				else         { bszids_pa = &bszids_ic[0]; \
							   thread_pa = thread_ic; } \
\
				/* Determine the packing buffer and related parameters for matrix
				   A. (If A will not be packed, then a_use will be set to point to
				   a and the _a_use strides will be set accordingly.) Then call
				   the packm sup variant chooser, which will call the appropriate
				   implementation based on the schema deduced from the stor_id. */ \
				PASTEMAC(ch,packm_sup_a) \
				( \
				  packa, \
				  BLIS_BUFFER_FOR_A_BLOCK, /* This algorithm packs matrix A to */ \
				  stor_id,                 /* a "block of A."                  */ \
				  BLIS_NO_TRANSPOSE, \
				  MC,     KC,       /* This "block of A" is (at most) MC x KC. */ \
				  mc_cur, kc_cur, MR, \
				  &one_local, \
				  a_ic,   rs_a,      cs_a, \
				  &a_use, &rs_a_use, &cs_a_use, \
				                     &ps_a_use, \
				  cntx, \
				  rntm, \
				  &mem_a, \
				  thread_pa  \
				); \
\
				/* Alias a_use so that it's clear this is our current block of
				   matrix A. */ \
				ctype* restrict a_ic_use = a_use; \
\
				/* Embed the panel stride of A within the auxinfo_t object. The
				   millikernel will query and use this to iterate through
				   micropanels of A (if needed). */ \
				bli_auxinfo_set_ps_a( ps_a_use, &aux ); \
\
				/* Grow the thrinfo_t tree. */ \
				bszid_t*   restrict bszids_jr = &bszids_pa[1]; \
				                    thread_jr = bli_thrinfo_sub_node( thread_pa ); \
				bli_thrinfo_sup_grow( rntm, bszids_jr, thread_jr ); \
\
				/* Compute number of primary and leftover components of the JR loop. */ \
				dim_t jr_iter = ( nc_pruned + NR - 1 ) / NR; \
				dim_t jr_left =   nc_pruned % NR; \
\
				/* Compute the JR loop thread range for the current thread. */ \
				dim_t jr_start, jr_end; \
				bli_thread_range_sub( thread_jr, jr_iter, 1, FALSE, &jr_start, &jr_end ); \
\
				/* An optimization: allow the last jr iteration to contain up to NRE
				   columns of C and B. (If NRE > NR, the mkernel has agreed to handle
				   these cases.) Note that this prevents us from declaring jr_iter and
				   jr_left as const. NOTE: We forgo this optimization when packing B
				   since packing an extended edge case is not yet supported. */ \
				if ( !packb && !is_mt ) \
				if ( NRE != 0 && 1 < jr_iter && jr_left != 0 && jr_left <= NRE ) \
				{ \
					jr_iter--; jr_left += NR; \
				} \
\
				/* Loop over the n dimension (NR columns at a time). */ \
				/*for ( dim_t j = 0; j < jr_iter; j += 1 )*/ \
				for ( dim_t j = jr_start; j < jr_end; j += 1 ) \
				{ \
					const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? NR : jr_left ); \
\
					/*
					ctype* restrict b_jr = b_pc_use + j * jrstep_b; \
					*/ \
					ctype* restrict b_jr = b_pc_use + j * ps_b_use; \
					ctype* restrict c_jr = c_ic     + j * jrstep_c; \
\
					dim_t i; \
					dim_t m_zero = 0; \
					dim_t n_iter_zero = 0; \
\
					m_off_cblock = m_off; \
					n_off_cblock = n_off + j * NR; \
\
					if(bli_gemmt_is_strictly_below_diag(m_off_cblock, n_off_cblock, mc_cur, nc_cur)) \
					{ \
						m_zero = 0; \
					} \
					else \
					{ \
						/* compute number of rows that are filled with zeroes and can be ignored */ \
						n_iter_zero = (n_off_cblock < m_off_cblock)? 0 : (n_off_cblock - m_off)/MR; \
						m_zero     = n_iter_zero * MR; \
					} \
\
					ctype* restrict a_ir = a_ic_use + n_iter_zero * ps_a_use; \
					ctype* restrict c_ir = c_jr + n_iter_zero * irstep_c; \
\
					/* Ignore the zero region */ \
					m_off_cblock += m_zero; \
\
					/* Compute the triangular part */ \
					for( i = m_zero; (i < mc_cur) && ( m_off_cblock < n_off_cblock + nr_cur); i += MR ) \
					{ \
						const dim_t mr_cur = (i+MR-1) < mc_cur ? MR : mc_cur - i; \
\
						gemmsup_ker \
						( \
						conja, \
						conjb, \
						mr_cur, \
						nr_cur, \
						kc_cur, \
						alpha_cast, \
						a_ir, rs_a_use, cs_a_use, \
						b_jr,     rs_b_use, cs_b_use, \
						zero, \
						ct,     rs_ct,     cs_ct, \
						&aux, \
						cntx  \
						); \
						if( col_pref ) \
						{ \
							PASTEMAC(ch,update_upper_triang)( n_off_cblock, m_off_cblock, \
							nr_cur, mr_cur, \
							ct, cs_ct, rs_ct, \
							beta_use, \
							c_ir, cs_c, rs_c ); \
						} \
						else \
						{ \
							PASTEMAC(ch,update_lower_triang)( m_off_cblock, n_off_cblock, \
							mr_cur, nr_cur, \
							ct, rs_ct, cs_ct, \
							beta_use, \
							c_ir, rs_c, cs_c ); \
						}\
\
						a_ir += ps_a_use; \
						c_ir += irstep_c; \
						m_off_cblock += mr_cur; \
					} \
\
					/* Invoke the gemmsup millikernel for remaining rectangular part. */ \
					gemmsup_ker \
					( \
					  conja, \
					  conjb, \
					  (i > mc_cur)? 0: mc_cur - i, \
					  nr_cur, \
					  kc_cur, \
					  alpha_cast, \
					  a_ir, rs_a_use, cs_a_use, \
					  b_jr,     rs_b_use, cs_b_use, \
					  beta_use, \
					  c_ir,     rs_c,     cs_c, \
					  &aux, \
					  cntx  \
					); \
\
				} \
			} \
\
			/* NOTE: This barrier is only needed if we are packing B (since
			   that matrix is packed within the pc loop of this variant). */ \
			if ( packb ) bli_thread_barrier( thread_pb ); \
		} \
	} \
\
	/* Release any memory that was acquired for packing matrices A and B. */ \
	PASTEMAC(ch,packm_sup_finalize_mem_a) \
	( \
	  packa, \
	  rntm, \
	  &mem_a, \
	  thread_pa  \
	); \
	PASTEMAC(ch,packm_sup_finalize_mem_b) \
	( \
	  packb, \
	  rntm, \
	  &mem_b, \
	  thread_pb  \
	); \
\
/*
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: b1", kc_cur, nr_cur, b_jr, rs_b, cs_b, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: a1", mr_cur, kc_cur, a_ir, rs_a, cs_a, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%4.1f", "" ); \
*/ \
}

INSERT_GENTFUNC_L_SC( gemmtsup, ref_var2m )

/* DGEMMT SUP kernel */
void bli_dgemmtsup_l_ref_var2m
     (
       bool             packa,
       bool             packb,
       conj_t           conja,
       conj_t           conjb,
       dim_t            m,
       dim_t            n,
       dim_t            k,
       void*   restrict alpha,
       void*   restrict a, inc_t rs_a, inc_t cs_a,
       void*   restrict b, inc_t rs_b, inc_t cs_b,
       void*   restrict beta,
       void*   restrict c, inc_t rs_c, inc_t cs_c,
       stor3_t          stor_id,
       cntx_t* restrict cntx,
       rntm_t* restrict rntm,
       thrinfo_t* restrict thread 
     )
{
	const num_t dt = PASTEMAC(d,type);

	double* restrict zero = PASTEMAC(d,0);

	/* If m or n is zero, return immediately. */
	if ( bli_zero_dim2( m, n ) ) return;

	/* If k < 1 or alpha is zero, scale by beta and return. */
	if ( k < 1 || PASTEMAC(d,eq0)( *(( double* )alpha) ) )
	{
		if ( bli_thread_am_ochief( thread ) )
		{
			PASTEMAC(d,scalm)
			(
			  BLIS_NO_CONJUGATE,
			  0,
			  BLIS_NONUNIT_DIAG,
			  BLIS_DENSE,
			  m, n,
			  beta,
			  c, rs_c, cs_c
			);
		}
		return;
	}

	/* Query the context for various blocksizes. */
	dim_t NR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NR, cntx );
	dim_t MR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MR, cntx );
	dim_t NC  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NC, cntx );
	dim_t MC  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MC, cntx );
	dim_t KC0 = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_KC, cntx );
	/* Query the maximum blocksize for NR, which implies a maximum blocksize
	   extension for the final iteration. */
	dim_t NRM = bli_cntx_get_l3_sup_tri_blksz_max_dt( dt, BLIS_NR, cntx );

	/* Query the context for the sup microkernel address and cast it to its
	   function pointer type. */
	PASTECH(d,gemmsup_ker_ft)
               gemmsup_ker = bli_cntx_get_l3_sup_tri_ker_dt( dt, stor_id, cntx );

	if( ( 0 == NR ) || ( 0 == MR ) || ( 0 == NC ) || ( 0 == MC ) || ( 0 == KC0 ) )
	{
		NR = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx );
		MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx );
		NC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx );
		MC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx );
		KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx );
		NRM = bli_cntx_get_l3_sup_blksz_max_dt( dt, BLIS_NR, cntx );
		gemmsup_ker = bli_cntx_get_l3_sup_ker_dt( dt, stor_id, cntx );
	}
	const dim_t NRE = NRM - NR;

	dim_t KC;
	if      ( packa && packb )
	{
		KC = KC0;
	}
	else if ( packb )
	{
		if      ( stor_id == BLIS_RRR ||
				  stor_id == BLIS_CCC    ) KC = KC0;
		else if ( stor_id == BLIS_RRC ||
				  stor_id == BLIS_CRC    ) KC = KC0;
		else if ( stor_id == BLIS_RCR ||
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4;
		else                               KC = KC0;
	}
	else if ( packa )
	{
		if      ( stor_id == BLIS_RRR ||
				  stor_id == BLIS_CCC    ) KC = (( KC0 / 2 ) / 2 ) * 2;
		else if ( stor_id == BLIS_RRC ||
				  stor_id == BLIS_CRC    ) KC = KC0;
		else if ( stor_id == BLIS_RCR ||
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4;
		else                               KC = KC0;
	}
	else /* if ( !packa && !packb ) */
	{
		if      ( stor_id == BLIS_RRR ||
				  stor_id == BLIS_CCC    ) KC = KC0;
		else if ( stor_id == BLIS_RRC ||
				  stor_id == BLIS_CRC    ) KC = KC0;
		else if ( m <=   MR && n <=   NR ) KC = KC0;
		else if ( m <= 2*MR && n <= 2*NR ) KC = KC0 / 2;
		else if ( m <= 3*MR && n <= 3*NR ) KC = (( KC0 / 3 ) / 4 ) * 4;
		else if ( m <= 4*MR && n <= 4*NR ) KC = KC0 / 4;
		else                               KC = (( KC0 / 5 ) / 4 ) * 4;
	}

	/* Compute partitioning step values for each matrix of each loop. */
	const inc_t jcstep_c = cs_c;
	const inc_t jcstep_b = cs_b;

	const inc_t pcstep_a = cs_a;
	const inc_t pcstep_b = rs_b;

	const inc_t icstep_c = rs_c;
	const inc_t icstep_a = rs_a;

	const inc_t jrstep_c = cs_c * NR;

	const inc_t irstep_c = rs_c * MR;

	/*
	const inc_t jrstep_b = cs_b * NR;
	( void )jrstep_b;

	const inc_t irstep_c = rs_c * MR;
	const inc_t irstep_a = rs_a * MR;
	*/

	double ct[ BLIS_STACK_BUF_MAX_SIZE / sizeof( double ) ]  __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE)));

	/* storage-scheme of ct should be same as that of C.
	  Since update routines only support row-major order,
	  col_pref flag is used to induce transpose to matrices before
	  passing to update routine whenever C is col-stored */
	const bool col_pref = (rs_c == 1)? 1 : 0;

	const inc_t rs_ct = ( col_pref ? 1 : NR );
	const inc_t cs_ct = ( col_pref ? MR : 1 );

	double* restrict a_00       = a;
	double* restrict b_00       = b;
	double* restrict c_00       = c;
	double* restrict alpha_cast = alpha;
	double* restrict beta_cast  = beta;

	/* Make local copies of beta and one scalars to prevent any unnecessary
	   sharing of cache lines between the cores' caches. */
	double           beta_local = *beta_cast;
	double           one_local  = *PASTEMAC(d,1);

	auxinfo_t       aux;

	/* Parse and interpret the contents of the rntm_t object to properly
	   set the ways of parallelism for each loop. */
	/*bli_rntm_set_ways_from_rntm_sup( m, n, k, rntm );*/

	/* Initialize a mem_t entry for A and B. Strictly speaking, this is only
	   needed for the matrix we will be packing (if any), but we do it
	   unconditionally to be safe. An alternative way of initializing the
	   mem_t entries is:

	     bli_mem_clear( &mem_a );
	     bli_mem_clear( &mem_b );
	*/
	mem_t mem_a = BLIS_MEM_INITIALIZER;
	mem_t mem_b = BLIS_MEM_INITIALIZER;

	/* Define an array of bszid_t ids, which will act as our substitute for
	   the cntl_t tree. */
	/*                           5thloop  4thloop         packb  3rdloop         packa  2ndloop  1stloop  ukrloop */
	bszid_t bszids_nopack[6] = { BLIS_NC, BLIS_KC,               BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t bszids_packa [7] = { BLIS_NC, BLIS_KC,               BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t bszids_packb [7] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t bszids_packab[8] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t* restrict bszids;

	/* Set the bszids pointer to the correct bszids array above based on which
	   matrices (if any) are being packed. */
	if ( packa ) { if ( packb ) bszids = bszids_packab;
	               else         bszids = bszids_packa; }
	else         { if ( packb ) bszids = bszids_packb;
	               else         bszids = bszids_nopack; }

	/* Determine whether we are using more than one thread. */
	const bool is_mt = bli_rntm_calc_num_threads( rntm );

	thrinfo_t* restrict thread_jc = NULL;
	thrinfo_t* restrict thread_pc = NULL;
	thrinfo_t* restrict thread_pb = NULL;
	thrinfo_t* restrict thread_ic = NULL;
	thrinfo_t* restrict thread_pa = NULL;
	thrinfo_t* restrict thread_jr = NULL;

	/* Grow the thrinfo_t tree. */
	bszid_t*   restrict bszids_jc = bszids;
	                    thread_jc = thread;
	bli_thrinfo_sup_grow( rntm, bszids_jc, thread_jc );

	/* Compute the JC loop thread range for the current thread. */
	dim_t jc_start, jc_end;
	bli_thread_range_weighted_sub( thread_jc, 0, BLIS_LOWER, m, n, NR, FALSE, &jc_start, &jc_end );
	const dim_t n_local = jc_end - jc_start;

	/* Compute number of primary and leftover components of the JC loop. */
	/*const dim_t jc_iter = ( n_local + NC - 1 ) / NC;*/
	const dim_t jc_left =   n_local % NC;

	dim_t m_off_cblock, n_off_cblock;
	dim_t m_off = 0;
	dim_t n_off = 0;
	doff_t diagoffc;
	dim_t i, ip;

	/* Loop over the n dimension (NC rows/columns at a time). */
	/*for ( dim_t jj = 0; jj < jc_iter; jj += 1 )*/
	for ( dim_t jj = jc_start; jj < jc_end; jj += NC )
	{
		/* Calculate the thread's current JC block dimension. */
		const dim_t nc_cur = ( NC <= jc_end - jj ? NC : jc_left );

		double* restrict b_jc = b_00 + jj * jcstep_b;
		double* restrict c_jc = c_00 + jj * jcstep_c;

		/* Grow the thrinfo_t tree. */
		bszid_t*   restrict bszids_pc = &bszids_jc[1];
		                    thread_pc = bli_thrinfo_sub_node( thread_jc );
		bli_thrinfo_sup_grow( rntm, bszids_pc, thread_pc );

		/* Compute the PC loop thread range for the current thread. */
		const dim_t pc_start = 0, pc_end = k;
		const dim_t k_local = k;

		/* Compute number of primary and leftover components of the PC loop. */
		/*const dim_t pc_iter = ( k_local + KC - 1 ) / KC;*/
		const dim_t pc_left =   k_local % KC;

		/* Loop over the k dimension (KC rows/columns at a time). */
		/*for ( dim_t pp = 0; pp < pc_iter; pp += 1 )*/
		for ( dim_t pp = pc_start; pp < pc_end; pp += KC )
		{
			/* Calculate the thread's current PC block dimension. */
			const dim_t kc_cur = ( KC <= pc_end - pp ? KC : pc_left );

			double* restrict a_pc = a_00 + pp * pcstep_a;
			double* restrict b_pc = b_jc + pp * pcstep_b;

			/* Only apply beta to the first iteration of the pc loop. */
			double* restrict beta_use = ( pp == 0 ? &beta_local : &one_local );

			m_off = 0;
			n_off = jj;
			diagoffc = m_off - n_off;

			double* b_use;
			inc_t  rs_b_use, cs_b_use, ps_b_use;

			/* Set the bszid_t array and thrinfo_t pointer based on whether
			   we will be packing B. If we won't be packing B, we alias to
			   the _pc variables so that code further down can unconditionally
			   reference the _pb variables. Note that *if* we will be packing
			   B, the thrinfo_t node will have already been created by a
			   previous call to bli_thrinfo_grow(), since bszid values of
			   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
			   bszid that is a normal bszid_t value). */
			bszid_t*   restrict bszids_pb;
			if ( packb ) { bszids_pb = &bszids_pc[1];
			               thread_pb = bli_thrinfo_sub_node( thread_pc ); }
			else         { bszids_pb = &bszids_pc[0];
			               thread_pb = thread_pc; }

			/* Determine the packing buffer and related parameters for matrix
			   B. (If B will not be packed, then a_use will be set to point to
			   b and the _b_use strides will be set accordingly.) Then call
			   the packm sup variant chooser, which will call the appropriate
			   implementation based on the schema deduced from the stor_id. */
			PASTEMAC(d,packm_sup_b)
			(
			  packb,
			  BLIS_BUFFER_FOR_B_PANEL, /* This algorithm packs matrix B to */
			  stor_id,                 /* a "panel of B."                  */
			  BLIS_NO_TRANSPOSE,
			  KC,     NC,       /* This "panel of B" is (at most) KC x NC. */
			  kc_cur, nc_cur, NR,
			  &one_local,
			  b_pc,   rs_b,      cs_b,
			  &b_use, &rs_b_use, &cs_b_use,
			                     &ps_b_use,
			  cntx,
			  rntm,
			  &mem_b,
			  thread_pb 
			);

			/* Alias a_use so that it's clear this is our current block of
			   matrix B. */
			double* restrict b_pc_use = b_use;

			/* We don't need to embed the panel stride of B within the auxinfo_t
			   object because this variant iterates through B in the jr loop,
			   which occurs here, within the macrokernel, not within the
			   millikernel. */
			/*bli_auxinfo_set_ps_b( ps_b_use, &aux );*/

			/* Grow the thrinfo_t tree. */
			bszid_t*   restrict bszids_ic = &bszids_pb[1];
			                    thread_ic = bli_thrinfo_sub_node( thread_pb );
			bli_thrinfo_sup_grow( rntm, bszids_ic, thread_ic );

			/* Compute the IC loop thread range for the current thread. */
			dim_t ic_start, ic_end;
			bli_thread_range_weighted_sub( thread_ic, -diagoffc, BLIS_UPPER, nc_cur, m, MR, FALSE, &ic_start, &ic_end );
			const dim_t m_local = ic_end - ic_start;

			/* Compute number of primary and leftover components of the IC loop. */
			/*const dim_t ic_iter = ( m_local + MC - 1 ) / MC;*/
			const dim_t ic_left =   m_local % MC;

			/* Loop over the m dimension (MC rows at a time). */
			/*for ( dim_t ii = 0; ii < ic_iter; ii += 1 )*/
			for ( dim_t ii = ic_start; ii < ic_end; ii += MC )
			{
				/* Calculate the thread's current IC block dimension. */
				dim_t mc_cur = ( MC <= ic_end - ii ? MC : ic_left );
				dim_t nc_pruned = nc_cur;

				double* restrict a_ic = a_pc + ii * icstep_a;
				double* restrict c_ic = c_jc + ii * icstep_c;

				m_off = ii;

				if(bli_gemmt_is_strictly_above_diag( m_off, n_off, mc_cur, nc_cur ) ) continue;

				diagoffc = m_off - n_off;

				if( diagoffc < 0 )
				{
					ip = -diagoffc / MR;
					i = ip * MR;
					mc_cur = mc_cur - i;
					diagoffc = -diagoffc % MR;
					m_off += i;
					c_ic = c_ic + ( i ) * rs_c;
					a_ic = a_ic + ( i ) * rs_a;
				}

				if( ( diagoffc + mc_cur ) < nc_cur )
				{
					nc_pruned = diagoffc + mc_cur;
				}

				double* a_use;
				inc_t  rs_a_use, cs_a_use, ps_a_use;

				/* Set the bszid_t array and thrinfo_t pointer based on whether
				   we will be packing B. If we won't be packing A, we alias to
				   the _ic variables so that code further down can unconditionally
				   reference the _pa variables. Note that *if* we will be packing
				   A, the thrinfo_t node will have already been created by a
				   previous call to bli_thrinfo_grow(), since bszid values of
				   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
				   bszid that is a normal bszid_t value). */
				bszid_t*   restrict bszids_pa;
				if ( packa ) { bszids_pa = &bszids_ic[1];
							   thread_pa = bli_thrinfo_sub_node( thread_ic ); }
				else         { bszids_pa = &bszids_ic[0];
							   thread_pa = thread_ic; }

				/* Determine the packing buffer and related parameters for matrix
				   A. (If A will not be packed, then a_use will be set to point to
				   a and the _a_use strides will be set accordingly.) Then call
				   the packm sup variant chooser, which will call the appropriate
				   implementation based on the schema deduced from the stor_id. */
				PASTEMAC(d,packm_sup_a)
				(
				  packa,
				  BLIS_BUFFER_FOR_A_BLOCK, /* This algorithm packs matrix A to */
				  stor_id,                 /* a "block of A."                  */
				  BLIS_NO_TRANSPOSE,
				  MC,     KC,       /* This "block of A" is (at most) MC x KC. */
				  mc_cur, kc_cur, MR,
				  &one_local,
				  a_ic,   rs_a,      cs_a,
				  &a_use, &rs_a_use, &cs_a_use,
				                     &ps_a_use,
				  cntx,
				  rntm,
				  &mem_a,
				  thread_pa 
				);

				/* Alias a_use so that it's clear this is our current block of
				   matrix A. */
				double* restrict a_ic_use = a_use;

				/* Embed the panel stride of A within the auxinfo_t object. The
				   millikernel will query and use this to iterate through
				   micropanels of A (if needed). */
				bli_auxinfo_set_ps_a( ps_a_use, &aux );

				/* Grow the thrinfo_t tree. */
				bszid_t*   restrict bszids_jr = &bszids_pa[1];
				                    thread_jr = bli_thrinfo_sub_node( thread_pa );
				bli_thrinfo_sup_grow( rntm, bszids_jr, thread_jr );

				/* Compute number of primary and leftover components of the JR loop. */
				dim_t jr_iter = ( nc_pruned + NR - 1 ) / NR;
				dim_t jr_left =   nc_pruned % NR;

				/* Compute the JR loop thread range for the current thread. */
				dim_t jr_start, jr_end;
				bli_thread_range_sub( thread_jr, jr_iter, 1, FALSE, &jr_start, &jr_end );

				/* An optimization: allow the last jr iteration to contain up to NRE
				   columns of C and B. (If NRE > NR, the mkernel has agreed to handle
				   these cases.) Note that this prevents us from declaring jr_iter and
				   jr_left as const. NOTE: We forgo this optimization when packing B
				   since packing an extended edge case is not yet supported. */
				if ( !packb && !is_mt )
				if ( NRE != 0 && 1 < jr_iter && jr_left != 0 && jr_left <= NRE )
				{
					jr_iter--; jr_left += NR;
				}

				/* Loop over the n dimension (NR columns at a time). */
				/*for ( dim_t j = 0; j < jr_iter; j += 1 )*/
				for ( dim_t j = jr_start; j < jr_end; j += 1 )
				{
					const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? NR : jr_left );

					/*
					double* restrict b_jr = b_pc_use + j * jrstep_b;
					*/
					double* restrict b_jr = b_pc_use + j * ps_b_use;
					double* restrict c_jr = c_ic     + j * jrstep_c;

					dim_t i;
					dim_t m_zero = 0;
					dim_t n_iter_zero = 0;

					m_off_cblock = m_off;
					n_off_cblock = n_off + j * NR;

					if(bli_gemmt_is_strictly_below_diag(m_off_cblock, n_off_cblock, mc_cur, nc_cur))
					{
						m_zero = 0;
					}
					else
					{
						/* compute number of rows that are filled with zeroes and can be ignored */
						n_iter_zero = (n_off_cblock < m_off_cblock)? 0 : (n_off_cblock - m_off)/MR;
						m_zero     = n_iter_zero * MR;
					}

					double* restrict a_ir = a_ic_use + n_iter_zero * ps_a_use;
					double* restrict c_ir = c_jr + n_iter_zero * irstep_c;

					/* Ignore the zero region */
					m_off_cblock += m_zero;

					/* Compute the triangular part */
					for( i = m_zero; (i < mc_cur) && ( m_off_cblock < n_off_cblock + nr_cur); i += MR )
					{
						const dim_t mr_cur = (i+MR-1) < mc_cur ? MR : mc_cur - i;
						dim_t m_off_24 = m_off_cblock % 24;
						dim_t n_off_24 = n_off_cblock % 24;
						dim_t m_idx = (dim_t)(m_off_24 / MR);
						dim_t n_idx = (dim_t)(n_off_24 / NR);
					#ifdef BLIS_KERNELS_ZEN4
						if ( (MR == 24) && (NR == 8) && bli_cpuid_is_avx512_supported() &&
								(stor_id != BLIS_CRC && stor_id != BLIS_RRC) &&
								// verify if micro panel intersects with diagonal
								// if distance from diagonal (n_off_cblock - m_off_cblock) is greater
								// than (LCM(MR, NR) - NR) then it implies that micro panel is far
								// from diagonal therefore it does not intersect with it.
								(n_off_cblock - m_off_cblock) <= 16 // (n_off_cblock - m_off_cblock) <= (LCM(MR, NR) - NR)
							)
						{
							/*
								call traingular 24x8 DGEMMT kernels
							*/
							// Difference between n_off_cblock and m_off_cblock is same as
							// the size of empty region before diagonal region.
							// kernel_idx = 0 is used when empty region size <= 0
							// kernel_idx = 1 is used when empty region size <= 8
							// kernel_idx = 2 is used when empty region size <= 16
							ker_fpls_zen4[(n_off_cblock - m_off_cblock)/NR]
							(
								conja,
								conjb,
								mr_cur,
								nr_cur,
								kc_cur,
								(double*) alpha_cast,
								(double*) a_ir, rs_a_use, cs_a_use,
								(double*) b_jr,     rs_b_use, cs_b_use,
								(double*) beta_use,
								(double*) c_ir,     rs_c,     cs_c,
								&aux,
								cntx
							);
							a_ir += ps_a_use;
							c_ir += irstep_c;
							m_off_cblock += mr_cur;
							continue;
						}
					#endif
					#ifdef BLIS_KERNELS_HASWELL
						/* Prerequisites : MR = 6, NR = 8.
						An optimization: allow the last jr iteration to contain up to NRE
						In DGEMMT API implementation, kernel operates on 6x8 block. MR and
						NR are set as 6 and 8 respectively. 24 being the LCM of 6 and 8,
						the diagonal pattern repeats for every 24x24 block.
						This pattern is exploited to achieve the optimization in diagonal
						blocks by computing only the required elements. In the previous
						implementation, all the 48 outputs of the given 6x8 block are
						computed and stored into a temporary buffer. Later, the required
						elements are copied into the final C output buffer.
						With this optimization, we are avoiding copy operation and also
						reducing the number of computations.
						Variables m_off_24 and n_off_24 respectively store the m and n
						offsets from the starting point of the corresponding 24x24 block.
						Variables m_idx and n_idx store indices of the current 6x8 block
						along m and n dimensions, in 24x24 block. m_idx is computed as
						(m_off_24 / MR) while n_idx is computed as (n_off_24 / NR).
						Range of m_idx is 0 <= m_idx <= 3 and the range of n_idx is
						0 <= n_idx <= 2. Based on these indices, for the given 6x8 block,
						logic is implemented to identify the relevant kernel from the
						look-up table.
						During instances, where m is not a multiple of 6 or n is not a
						multiple of 8, it goes to the default gemm kernel. MR and NR must be
						6 and 8 for these kernels to achieve the expected functionality.*/


						/* Check if m, n indices are multiple of MR and NR respectively
							and current block is a complete 6x8 block */
						bool idx_supported = ((m_off_24 % MR) == 0) && ((n_off_24 % NR) == 0)
						&& (MR == 6) && (NR == 8)
						&& (bli_cpuid_is_avx2fma3_supported() == TRUE) && (mr_cur == MR) && (nr_cur == NR);

						/* m_idx and n_idx would be equal only if the current block is
							a diagonal block */
						if( (dt == BLIS_DOUBLE) && (m_idx == n_idx) && (idx_supported) )
						{
							/* index of kernel in lookup table is 2*m_idx) */
							dim_t ker_idx;
							ker_idx = m_idx<<1;

							/* If there is another 6x8 diagonal block pending for computation
								after the current 6x8 diagonal block, then the two blocks can
								be computed together(12x8). This combined kernel is implemented
								only for the case where n_idx = 2 i.e., n_off_24 = 16. To call
								this, it has to be ensured that at least 12 rows are pending in
								C for computation. (m_off + 2 * MR <=m). Usage of this combined
								kernel saves the entire time to execute one kernel*/
							if( (n_idx == 2) && (m_off_cblock + MR + MR <= m) ) {
								ker_idx = 6; /* use combined kernel, index of combined kernel
												in lookup table is 6 */
							}
							/* use rd kernel if B is column major storage */
							if( stor_id == BLIS_RRC ) {
								ker_idx += 7; /* index of rd kernel*/
							}
							gemmt_ker_ft ker_fp = ker_fpls_haswell[ker_idx];
							ker_fp
							(
								conja,
								conjb,
								mr_cur,
								nr_cur,
								kc_cur,
								(double*) alpha_cast,
								(double*) a_ir, rs_a_use, cs_a_use,
								(double*) b_jr,     rs_b_use, cs_b_use,
								(double*) beta_use,
								(double*) c_ir,     rs_c,     cs_c,
								&aux,
								cntx 
							);
							a_ir += ps_a_use;
							c_ir += irstep_c;
							m_off_cblock += mr_cur;
							continue;
						}
						/* 6x8 block where m_idx == n_idx+1 also has some parts of the diagonal */
						else if ( (dt == BLIS_DOUBLE) && (m_idx == n_idx+1) && (idx_supported) ) 
						{
							/* If current block was already computed in the combined kernel it
							   can be skipped combined kernel is only implemented for n_idx=2,
							   i == m_zero is only true for the first iteration therefore if
							   i == m_zero then the current 6x8 block was not computed in
							   combined kernel
							*/
							if ((n_idx != 2) || (i == m_zero))
							{
								dim_t ker_idx = (n_idx << 1) + 1;
								/* use rd kernel if B is column major storage */
								if( stor_id == BLIS_RRC ) { ker_idx += 7; }
								gemmt_ker_ft ker_fp = ker_fpls_haswell[ker_idx];
								ker_fp
								(
									conja,
									conjb,
									mr_cur,
									nr_cur,
									kc_cur,
									(double*) alpha_cast,
									(double*) a_ir, rs_a_use, cs_a_use,
									(double*) b_jr,     rs_b_use, cs_b_use,
									(double*) beta_use,
									(double*) c_ir,     rs_c,     cs_c,
									&aux,
									cntx
								);
							}
							a_ir += ps_a_use;
							c_ir += irstep_c;
							m_off_cblock += mr_cur;
							continue;
						}
					#endif
						gemmsup_ker
						(
						conja,
						conjb,
						mr_cur,
						nr_cur,
						kc_cur,
						alpha_cast,
						a_ir, rs_a_use, cs_a_use,
						b_jr,     rs_b_use, cs_b_use,
						zero,
						ct,     rs_ct,     cs_ct,
						&aux,
						cntx 
						);
						if( col_pref )
						{
							PASTEMAC(d,update_upper_triang)( n_off_cblock, m_off_cblock,
							nr_cur, mr_cur,
							ct, cs_ct, rs_ct,
							beta_use,
							c_ir, cs_c, rs_c );
						}
						else
						{
							PASTEMAC(d,update_lower_triang)( m_off_cblock, n_off_cblock,
							mr_cur, nr_cur,
							ct, rs_ct, cs_ct,
							beta_use,
							c_ir, rs_c, cs_c );
						}

						a_ir += ps_a_use;
						c_ir += irstep_c;
						m_off_cblock += mr_cur;
					}

					/* Invoke the gemmsup millikernel for remaining rectangular part. */
					gemmsup_ker
					(
					  conja,
					  conjb,
					  (i > mc_cur)? 0: mc_cur - i,
					  nr_cur,
					  kc_cur,
					  alpha_cast,
					  a_ir, rs_a_use, cs_a_use,
					  b_jr,     rs_b_use, cs_b_use,
					  beta_use,
					  c_ir,     rs_c,     cs_c,
					  &aux,
					  cntx 
					);

				}
			}

			/* NOTE: This barrier is only needed if we are packing B (since
			   that matrix is packed within the pc loop of this variant). */
			if ( packb ) bli_thread_barrier( thread_pb );
		}
	}

	/* Release any memory that was acquired for packing matrices A and B. */
	PASTEMAC(d,packm_sup_finalize_mem_a)
	(
	  packa,
	  rntm,
	  &mem_a,
	  thread_pa 
	);
	PASTEMAC(d,packm_sup_finalize_mem_b)
	(
	  packb,
	  rntm,
	  &mem_b,
	  thread_pb 
	);

/*
PASTEMAC(d,fprintm)( stdout, "gemmsup_ref_var2: b1", kc_cur, nr_cur, b_jr, rs_b, cs_b, "%4.1f", "" );
PASTEMAC(d,fprintm)( stdout, "gemmsup_ref_var2: a1", mr_cur, kc_cur, a_ir, rs_a, cs_a, "%4.1f", "" );
PASTEMAC(d,fprintm)( stdout, "gemmsup_ref_var2: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%4.1f", "" );
*/
}

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, uplo, varname ) \
\
void PASTEMACT(ch,opname,uplo,varname) \
     ( \
       bool             packa, \
       bool             packb, \
       conj_t           conja, \
       conj_t           conjb, \
       dim_t            m, \
       dim_t            n, \
       dim_t            k, \
       void*   restrict alpha, \
       void*   restrict a, inc_t rs_a, inc_t cs_a, \
       void*   restrict b, inc_t rs_b, inc_t cs_b, \
       void*   restrict beta, \
       void*   restrict c, inc_t rs_c, inc_t cs_c, \
       stor3_t          stor_id, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype* restrict zero = PASTEMAC(ch,0); \
\
	/* If m or n is zero, return immediately. */ \
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* If k < 1 or alpha is zero, scale by beta and return. */ \
	if ( k < 1 || PASTEMAC(ch,eq0)( *(( ctype* )alpha) ) ) \
	{ \
		if ( bli_thread_am_ochief( thread ) ) \
		{ \
			PASTEMAC(ch,scalm) \
			( \
			  BLIS_NO_CONJUGATE, \
			  0, \
			  BLIS_NONUNIT_DIAG, \
			  BLIS_DENSE, \
			  m, n, \
			  beta, \
			  c, rs_c, cs_c \
			); \
		} \
		return; \
	} \
\
	/* Query the context for various blocksizes. */ \
	dim_t NR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NR, cntx ); \
	dim_t MR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MR, cntx ); \
	dim_t NC  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NC, cntx ); \
	dim_t MC  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MC, cntx ); \
	dim_t KC0 = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_KC, cntx ); \
\
	/* Query the maximum blocksize for NR, which implies a maximum blocksize
	   extension for the final iteration. */ \
	dim_t NRM = bli_cntx_get_l3_sup_tri_blksz_max_dt( dt, BLIS_NR, cntx ); \
\
	/* Query the context for the sup microkernel address and cast it to its
	   function pointer type. */ \
	PASTECH(ch,gemmsup_ker_ft) \
               gemmsup_ker = bli_cntx_get_l3_sup_tri_ker_dt( dt, stor_id, cntx ); \
\
	if( ( 0 == NR ) || ( 0 == MR ) || ( 0 == NC ) || ( 0 == MC ) || ( 0 == KC0 ) ) \
	{ \
		NR = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx ); \
		MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx ); \
		NC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx ); \
		MC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx ); \
		KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx ); \
		NRM = bli_cntx_get_l3_sup_blksz_max_dt( dt, BLIS_NR, cntx ); \
		gemmsup_ker = bli_cntx_get_l3_sup_ker_dt( dt, stor_id, cntx ); \
	} \
	const dim_t NRE = NRM - NR; \
\
	dim_t KC; \
	if      ( packa && packb ) \
	{ \
		KC = KC0; \
	} \
	else if ( packb ) \
	{ \
		if      ( stor_id == BLIS_RRR || \
				  stor_id == BLIS_CCC    ) KC = KC0; \
		else if ( stor_id == BLIS_RRC || \
				  stor_id == BLIS_CRC    ) KC = KC0; \
		else if ( stor_id == BLIS_RCR || \
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4; \
		else                               KC = KC0; \
	} \
	else if ( packa ) \
	{ \
		if      ( stor_id == BLIS_RRR || \
				  stor_id == BLIS_CCC    ) KC = (( KC0 / 2 ) / 2 ) * 2; \
		else if ( stor_id == BLIS_RRC || \
				  stor_id == BLIS_CRC    ) KC = KC0; \
		else if ( stor_id == BLIS_RCR || \
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4; \
		else                               KC = KC0; \
	} \
	else /* if ( !packa && !packb ) */ \
	{ \
		if      ( stor_id == BLIS_RRR || \
				  stor_id == BLIS_CCC    ) KC = KC0; \
		else if ( stor_id == BLIS_RRC || \
				  stor_id == BLIS_CRC    ) KC = KC0; \
		else if ( stor_id == BLIS_RCR ) \
		{ \
		     if      ( m <=  4*MR ) KC = KC0; \
		     else if ( m <= 36*MR ) KC = KC0 / 2; \
		     else if ( m <= 56*MR ) KC = (( KC0 / 3 ) / 4 ) * 4; \
		     else                   KC = KC0 / 4; \
		} \
		else if ( m <=   MR && n <=   NR ) KC = KC0; \
		else if ( m <= 2*MR && n <= 2*NR ) KC = KC0 / 2; \
		else if ( m <= 3*MR && n <= 3*NR ) KC = (( KC0 / 3 ) / 4 ) * 4; \
		else if ( m <= 4*MR && n <= 4*NR ) KC = KC0 / 4; \
		else                               KC = (( KC0 / 5 ) / 4 ) * 4; \
	} \
\
	/* Compute partitioning step values for each matrix of each loop. */ \
	const inc_t jcstep_c = cs_c; \
	const inc_t jcstep_b = cs_b; \
\
	const inc_t pcstep_a = cs_a; \
	const inc_t pcstep_b = rs_b; \
\
	const inc_t icstep_c = rs_c; \
	const inc_t icstep_a = rs_a; \
\
	const inc_t jrstep_c = cs_c * NR; \
\
	const inc_t irstep_c = rs_c * MR; \
\
	/*
	const inc_t jrstep_b = cs_b * NR; \
	( void )jrstep_b; \
\
	const inc_t irstep_c = rs_c * MR; \
	const inc_t irstep_a = rs_a * MR; \
	*/ \
\
	ctype ct[ BLIS_STACK_BUF_MAX_SIZE / sizeof( ctype ) ] __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
\
	/* Storage scheme of ct should be same as that of C.
	   Since update routines only support row-major order,
	   col_pref flag is used to induce transpose to matrices before
	   passing to update routine whenever C is col-stored */ \
	const bool col_pref = (rs_c == 1) ? 1 : 0; \
\
	const inc_t rs_ct = ( col_pref ? 1 : NR ); \
	const inc_t cs_ct = ( col_pref ? MR : 1 ); \
\
	ctype* restrict a_00       = a; \
	ctype* restrict b_00       = b; \
	ctype* restrict c_00       = c; \
	ctype* restrict alpha_cast = alpha; \
	ctype* restrict beta_cast  = beta; \
\
	/* Make local copies of beta and one scalars to prevent any unnecessary
	   sharing of cache lines between the cores' caches. */ \
	ctype           beta_local = *beta_cast; \
	ctype           one_local  = *PASTEMAC(ch,1); \
\
	auxinfo_t       aux; \
\
	/* Parse and interpret the contents of the rntm_t object to properly
	   set the ways of parallelism for each loop. */ \
	/*bli_rntm_set_ways_from_rntm_sup( m, n, k, rntm );*/ \
\
	/* Initialize a mem_t entry for A and B. Strictly speaking, this is only
	   needed for the matrix we will be packing (if any), but we do it
	   unconditionally to be safe. An alternative way of initializing the
	   mem_t entries is:

	     bli_mem_clear( &mem_a ); \
	     bli_mem_clear( &mem_b ); \
	*/ \
	mem_t mem_a = BLIS_MEM_INITIALIZER; \
	mem_t mem_b = BLIS_MEM_INITIALIZER; \
\
	/* Define an array of bszid_t ids, which will act as our substitute for
	   the cntl_t tree. */ \
	/*                           5thloop  4thloop         packb  3rdloop         packa  2ndloop  1stloop  ukrloop */ \
	bszid_t bszids_nopack[6] = { BLIS_NC, BLIS_KC,               BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t bszids_packa [7] = { BLIS_NC, BLIS_KC,               BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t bszids_packb [7] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t bszids_packab[8] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t* restrict bszids; \
\
	/* Set the bszids pointer to the correct bszids array above based on which
	   matrices (if any) are being packed. */ \
	if ( packa ) { if ( packb ) bszids = bszids_packab; \
	               else         bszids = bszids_packa; } \
	else         { if ( packb ) bszids = bszids_packb; \
	               else         bszids = bszids_nopack; } \
\
	/* Determine whether we are using more than one thread. */ \
	const bool is_mt = bli_rntm_calc_num_threads( rntm ); \
\
	thrinfo_t* restrict thread_jc = NULL; \
	thrinfo_t* restrict thread_pc = NULL; \
	thrinfo_t* restrict thread_pb = NULL; \
	thrinfo_t* restrict thread_ic = NULL; \
	thrinfo_t* restrict thread_pa = NULL; \
	thrinfo_t* restrict thread_jr = NULL; \
\
	/* Grow the thrinfo_t tree. */ \
	bszid_t*   restrict bszids_jc = bszids; \
	                    thread_jc = thread; \
	bli_thrinfo_sup_grow( rntm, bszids_jc, thread_jc ); \
\
	/* Compute the JC loop thread range for the current thread. */ \
	dim_t jc_start, jc_end; \
	bli_thread_range_weighted_sub( thread_jc, 0, BLIS_UPPER, m, n, NR, FALSE, &jc_start, &jc_end ); \
	const dim_t n_local = jc_end - jc_start; \
\
	dim_t m_off = 0; \
	dim_t n_off = 0; \
	doff_t diagoffc; \
	dim_t m_off_cblock, n_off_cblock; \
	dim_t jp, j; \
\
	/* Compute number of primary and leftover components of the JC loop. */ \
	/*const dim_t jc_iter = ( n_local + NC - 1 ) / NC;*/ \
	const dim_t jc_left =   n_local % NC; \
\
	/* Loop over the n dimension (NC rows/columns at a time). */ \
	/*for ( dim_t jj = 0; jj < jc_iter; jj += 1 )*/ \
	for ( dim_t jj = jc_start; jj < jc_end; jj += NC ) \
	{ \
		/* Calculate the thread's current JC block dimension. */ \
		const dim_t nc_cur = ( NC <= jc_end - jj ? NC : jc_left ); \
\
		ctype* restrict b_jc = b_00 + jj * jcstep_b; \
		ctype* restrict c_jc = c_00 + jj * jcstep_c; \
\
		/* Grow the thrinfo_t tree. */ \
		bszid_t*   restrict bszids_pc = &bszids_jc[1]; \
		                    thread_pc = bli_thrinfo_sub_node( thread_jc ); \
		bli_thrinfo_sup_grow( rntm, bszids_pc, thread_pc ); \
\
		/* Compute the PC loop thread range for the current thread. */ \
		const dim_t pc_start = 0, pc_end = k; \
		const dim_t k_local = k; \
\
		/* Compute number of primary and leftover components of the PC loop. */ \
		/*const dim_t pc_iter = ( k_local + KC - 1 ) / KC;*/ \
		const dim_t pc_left =   k_local % KC; \
\
		/* Loop over the k dimension (KC rows/columns at a time). */ \
		/*for ( dim_t pp = 0; pp < pc_iter; pp += 1 )*/ \
		for ( dim_t pp = pc_start; pp < pc_end; pp += KC ) \
		{ \
			/* Calculate the thread's current PC block dimension. */ \
			const dim_t kc_cur = ( KC <= pc_end - pp ? KC : pc_left ); \
\
			ctype* restrict a_pc = a_00 + pp * pcstep_a; \
			ctype* restrict b_pc = b_jc + pp * pcstep_b; \
\
			/* Only apply beta to the first iteration of the pc loop. */ \
			ctype* restrict beta_use = ( pp == 0 ? &beta_local : &one_local ); \
\
			m_off = 0; \
			n_off = jj; \
			diagoffc = m_off - n_off; \
\
			ctype* b_use; \
			inc_t  rs_b_use, cs_b_use, ps_b_use; \
\
			/* Set the bszid_t array and thrinfo_t pointer based on whether
			   we will be packing B. If we won't be packing B, we alias to
			   the _pc variables so that code further down can unconditionally
			   reference the _pb variables. Note that *if* we will be packing
			   B, the thrinfo_t node will have already been created by a
			   previous call to bli_thrinfo_grow(), since bszid values of
			   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
			   bszid that is a normal bszid_t value). */ \
			bszid_t*   restrict bszids_pb; \
			if ( packb ) { bszids_pb = &bszids_pc[1]; \
			               thread_pb = bli_thrinfo_sub_node( thread_pc ); } \
			else         { bszids_pb = &bszids_pc[0]; \
			               thread_pb = thread_pc; } \
\
			/* Determine the packing buffer and related parameters for matrix
			   B. (If B will not be packed, then a_use will be set to point to
			   b and the _b_use strides will be set accordingly.) Then call
			   the packm sup variant chooser, which will call the appropriate
			   implementation based on the schema deduced from the stor_id. */ \
			PASTEMAC(ch,packm_sup_b) \
			( \
			  packb, \
			  BLIS_BUFFER_FOR_B_PANEL, /* This algorithm packs matrix B to */ \
			  stor_id,                 /* a "panel of B."                  */ \
			  BLIS_NO_TRANSPOSE, \
			  KC,     NC,       /* This "panel of B" is (at most) KC x NC. */ \
			  kc_cur, nc_cur, NR, \
			  &one_local, \
			  b_pc,   rs_b,      cs_b, \
			  &b_use, &rs_b_use, &cs_b_use, \
			                     &ps_b_use, \
			  cntx, \
			  rntm, \
			  &mem_b, \
			  thread_pb  \
			); \
\
			/* Alias a_use so that it's clear this is our current block of
			   matrix B. */ \
			ctype* restrict b_pc_use = b_use; \
\
			/* We don't need to embed the panel stride of B within the auxinfo_t
			   object because this variant iterates through B in the jr loop,
			   which occurs here, within the macrokernel, not within the
			   millikernel. */ \
			/*bli_auxinfo_set_ps_b( ps_b_use, &aux );*/ \
\
			/* Grow the thrinfo_t tree. */ \
			bszid_t*   restrict bszids_ic = &bszids_pb[1]; \
			                    thread_ic = bli_thrinfo_sub_node( thread_pb ); \
			bli_thrinfo_sup_grow( rntm, bszids_ic, thread_ic ); \
\
			/* Compute the IC loop thread range for the current thread. */ \
			dim_t ic_start, ic_end; \
			bli_thread_range_weighted_sub( thread_ic, -diagoffc, BLIS_LOWER, nc_cur, m, MR, FALSE, &ic_start, &ic_end ); \
			const dim_t m_local = ic_end - ic_start; \
\
			/* Compute number of primary and leftover components of the IC loop. */ \
			/*const dim_t ic_iter = ( m_local + MC - 1 ) / MC;*/ \
			const dim_t ic_left =   m_local % MC; \
\
			/* Loop over the m dimension (MC rows at a time). */ \
			/*for ( dim_t ii = 0; ii < ic_iter; ii += 1 )*/ \
			for ( dim_t ii = ic_start; ii < ic_end; ii += MC ) \
			{ \
				/* Calculate the thread's current IC block dimension. */ \
				dim_t mc_cur = ( MC <= ic_end - ii ? MC : ic_left ); \
\
				dim_t nc_pruned = nc_cur; \
\
				m_off = ii; \
				n_off = jj; \
\
				if(bli_gemmt_is_strictly_below_diag(m_off, n_off, mc_cur, nc_cur)) continue; \
\
				ctype* restrict a_ic = a_pc + ii * icstep_a; \
				ctype* restrict c_ic = c_jc + ii * icstep_c; \
\
				doff_t diagoffc = m_off - n_off; \
\
				ctype* restrict b_pc_pruned = b_pc_use; \
\
				if(diagoffc > 0 ) \
				{ \
					jp = diagoffc / NR; \
					j = jp * NR; \
					nc_pruned = nc_cur - j; \
					n_off += j; \
					diagoffc = diagoffc % NR; \
					c_ic = c_ic + ( j ) * cs_c; \
					b_pc_pruned = b_pc_use + ( jp ) * ps_b_use; \
				} \
\
				if( ( ( -diagoffc ) + nc_pruned ) < mc_cur ) \
				{ \
					mc_cur = -diagoffc + nc_pruned; \
				} \
\
				ctype* a_use; \
				inc_t  rs_a_use, cs_a_use, ps_a_use; \
\
				/* Set the bszid_t array and thrinfo_t pointer based on whether
				   we will be packing B. If we won't be packing A, we alias to
				   the _ic variables so that code further down can unconditionally
				   reference the _pa variables. Note that *if* we will be packing
				   A, the thrinfo_t node will have already been created by a
				   previous call to bli_thrinfo_grow(), since bszid values of
				   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
				   bszid that is a normal bszid_t value). */ \
				bszid_t*   restrict bszids_pa; \
				if ( packa ) { bszids_pa = &bszids_ic[1]; \
							   thread_pa = bli_thrinfo_sub_node( thread_ic ); } \
				else         { bszids_pa = &bszids_ic[0]; \
							   thread_pa = thread_ic; } \
\
				/* Determine the packing buffer and related parameters for matrix
				   A. (If A will not be packed, then a_use will be set to point to
				   a and the _a_use strides will be set accordingly.) Then call
				   the packm sup variant chooser, which will call the appropriate
				   implementation based on the schema deduced from the stor_id. */ \
				PASTEMAC(ch,packm_sup_a) \
				( \
				  packa, \
				  BLIS_BUFFER_FOR_A_BLOCK, /* This algorithm packs matrix A to */ \
				  stor_id,                 /* a "block of A."                  */ \
				  BLIS_NO_TRANSPOSE, \
				  MC,     KC,       /* This "block of A" is (at most) MC x KC. */ \
				  mc_cur, kc_cur, MR, \
				  &one_local, \
				  a_ic,   rs_a,      cs_a, \
				  &a_use, &rs_a_use, &cs_a_use, \
				                     &ps_a_use, \
				  cntx, \
				  rntm, \
				  &mem_a, \
				  thread_pa  \
				); \
\
				/* Alias a_use so that it's clear this is our current block of
				   matrix A. */ \
				ctype* restrict a_ic_use = a_use; \
\
				/* Embed the panel stride of A within the auxinfo_t object. The
				   millikernel will query and use this to iterate through
				   micropanels of A (if needed). */ \
				bli_auxinfo_set_ps_a( ps_a_use, &aux ); \
\
				/* Grow the thrinfo_t tree. */ \
				bszid_t*   restrict bszids_jr = &bszids_pa[1]; \
				                    thread_jr = bli_thrinfo_sub_node( thread_pa ); \
				bli_thrinfo_sup_grow( rntm, bszids_jr, thread_jr ); \
\
				/* Compute number of primary and leftover components of the JR loop. */ \
				dim_t jr_iter = ( nc_pruned + NR - 1 ) / NR; \
				dim_t jr_left =   nc_pruned % NR; \
\
				/* Compute the JR loop thread range for the current thread. */ \
				dim_t jr_start, jr_end; \
				bli_thread_range_sub( thread_jr, jr_iter, 1, FALSE, &jr_start, &jr_end ); \
\
				/* An optimization: allow the last jr iteration to contain up to NRE
				   columns of C and B. (If NRE > NR, the mkernel has agreed to handle
				   these cases.) Note that this prevents us from declaring jr_iter and
				   jr_left as const. NOTE: We forgo this optimization when packing B
				   since packing an extended edge case is not yet supported. */ \
				if ( !packb && !is_mt ) \
				if ( NRE != 0 && 1 < jr_iter && jr_left != 0 && jr_left <= NRE ) \
				{ \
					jr_iter--; jr_left += NR; \
				} \
\
				/* Loop over the n dimension (NR columns at a time). */ \
				/*for ( dim_t j = 0; j < jr_iter; j += 1 )*/ \
				for ( dim_t j = jr_start; j < jr_end; j += 1 ) \
				{ \
					const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? NR : jr_left ); \
\
					/*
					ctype* restrict b_jr = b_pc_use + j * jrstep_b; \
					*/ \
					ctype* restrict b_jr = b_pc_pruned + j * ps_b_use; \
					ctype* restrict c_jr = c_ic     + j * jrstep_c; \
					dim_t m_rect = 0; \
				        dim_t n_iter_rect = 0; \
\
					m_off_cblock = m_off; \
					n_off_cblock = n_off + j * NR; \
\
					if(bli_gemmt_is_strictly_above_diag(m_off_cblock, n_off_cblock, mc_cur, nr_cur)) \
					{ \
						m_rect = mc_cur; \
					} \
					else \
					{ \
						/* calculate the number of rows in rectangular region of the block */ \
						n_iter_rect = n_off_cblock < m_off_cblock ? 0: (n_off_cblock - m_off_cblock) / MR; \
						m_rect = n_iter_rect * MR; \
					} \
\
					/* Compute the rectangular part */ \
					gemmsup_ker \
					( \
					  conja, \
					  conjb, \
					  m_rect, \
					  nr_cur, \
					  kc_cur, \
					  alpha_cast, \
					  a_ic_use, rs_a_use, cs_a_use, \
					  b_jr,     rs_b_use, cs_b_use, \
					  beta_use, \
					  c_jr,     rs_c,     cs_c, \
					  &aux, \
					  cntx  \
					); \
\
					m_off_cblock = m_off + m_rect; \
\
					ctype* restrict a_ir = a_ic_use + n_iter_rect * ps_a_use; \
					ctype* restrict c_ir = c_jr + n_iter_rect * irstep_c; \
\
					/* compute the remaining triangular part */ \
					for( dim_t i = m_rect;( i < mc_cur) && (m_off_cblock < n_off_cblock + nr_cur); i += MR ) \
					{ \
						const dim_t mr_cur = (i+MR-1) < mc_cur ? MR : mc_cur - i; \
						{ \
							gemmsup_ker \
							( \
							conja, \
							conjb, \
							mr_cur, \
							nr_cur, \
							kc_cur, \
							alpha_cast, \
							a_ir, rs_a_use, cs_a_use, \
							b_jr,     rs_b_use, cs_b_use, \
							zero, \
							ct,     rs_ct,     cs_ct,  \
							&aux, \
							cntx  \
							); \
	\
							if( col_pref ) \
							{ \
								PASTEMAC(ch,update_lower_triang)( n_off_cblock, m_off_cblock,  \
									nr_cur, mr_cur, \
									ct, cs_ct, rs_ct, \
									beta_use, \
									c_ir, cs_c, rs_c ); \
							} \
							else \
							{ \
								PASTEMAC(ch,update_upper_triang)( m_off_cblock, n_off_cblock,  \
									mr_cur, nr_cur, \
									ct, rs_ct, cs_ct, \
									beta_use, \
									c_ir, rs_c, cs_c ); \
							} \
						} \
\
						a_ir += ps_a_use; \
						c_ir += irstep_c; \
						m_off_cblock += mr_cur; \
\
					} \
				} \
			} \
\
			/* NOTE: This barrier is only needed if we are packing B (since
			   that matrix is packed within the pc loop of this variant). */ \
			if ( packb ) bli_thread_barrier( thread_pb ); \
		} \
	} \
\
	/* Release any memory that was acquired for packing matrices A and B. */ \
	PASTEMAC(ch,packm_sup_finalize_mem_a) \
	( \
	  packa, \
	  rntm, \
	  &mem_a, \
	  thread_pa  \
	); \
	PASTEMAC(ch,packm_sup_finalize_mem_b) \
	( \
	  packb, \
	  rntm, \
	  &mem_b, \
	  thread_pb  \
	); \
\
/*
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: b1", kc_cur, nr_cur, b_jr, rs_b, cs_b, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: a1", mr_cur, kc_cur, a_ir, rs_a, cs_a, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%4.1f", "" ); \
*/ \
}

INSERT_GENTFUNC_U_SC( gemmtsup, ref_var2m )

void bli_dgemmtsup_u_ref_var2m
     (
       bool             packa,
       bool             packb,
       conj_t           conja,
       conj_t           conjb,
       dim_t            m,
       dim_t            n,
       dim_t            k,
       void*   restrict alpha,
       void*   restrict a, inc_t rs_a, inc_t cs_a,
       void*   restrict b, inc_t rs_b, inc_t cs_b,
       void*   restrict beta,
       void*   restrict c, inc_t rs_c, inc_t cs_c,
       stor3_t          stor_id,
       cntx_t* restrict cntx,
       rntm_t* restrict rntm,
       thrinfo_t* restrict thread 
     )
{
	const num_t dt = PASTEMAC(d,type);

	double* restrict zero = PASTEMAC(d,0);

	/* If m or n is zero, return immediately. */
	if ( bli_zero_dim2( m, n ) ) return;

	/* If k < 1 or alpha is zero, scale by beta and return. */
	if ( k < 1 || PASTEMAC(d,eq0)( *(( double* )alpha) ) )
	{
		if ( bli_thread_am_ochief( thread ) )
		{
			PASTEMAC(d,scalm)
			(
			  BLIS_NO_CONJUGATE,
			  0,
			  BLIS_NONUNIT_DIAG,
			  BLIS_DENSE,
			  m, n,
			  beta,
			  c, rs_c, cs_c
			);
		}
		return;
	}

	/* Query the context for various blocksizes. */
	dim_t NR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NR, cntx );
	dim_t MR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MR, cntx );
	dim_t NC  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NC, cntx );
	dim_t MC  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MC, cntx );
	dim_t KC0 = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_KC, cntx );

	/* Query the maximum blocksize for NR, which implies a maximum blocksize
	   extension for the final iteration. */
	dim_t NRM = bli_cntx_get_l3_sup_tri_blksz_max_dt( dt, BLIS_NR, cntx );

	/* Query the context for the sup microkernel address and cast it to its
	   function pointer type. */
	PASTECH(d,gemmsup_ker_ft)
               gemmsup_ker = bli_cntx_get_l3_sup_tri_ker_dt( dt, stor_id, cntx );

	if( ( 0 == NR ) || ( 0 == MR ) || ( 0 == NC ) || ( 0 == MC ) || ( 0 == KC0 ) )
	{
		NR = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx );
		MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx );
		NC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx );
		MC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx );
		KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx );
		NRM = bli_cntx_get_l3_sup_blksz_max_dt( dt, BLIS_NR, cntx );
		gemmsup_ker = bli_cntx_get_l3_sup_ker_dt( dt, stor_id, cntx );
	}
	const dim_t NRE = NRM - NR;

	dim_t KC;
	if      ( packa && packb )
	{
		KC = KC0;
	}
	else if ( packb )
	{
		if      ( stor_id == BLIS_RRR ||
				  stor_id == BLIS_CCC    ) KC = KC0;
		else if ( stor_id == BLIS_RRC ||
				  stor_id == BLIS_CRC    ) KC = KC0;
		else if ( stor_id == BLIS_RCR ||
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4;
		else                               KC = KC0;
	}
	else if ( packa )
	{
		if      ( stor_id == BLIS_RRR ||
				  stor_id == BLIS_CCC    ) KC = (( KC0 / 2 ) / 2 ) * 2;
		else if ( stor_id == BLIS_RRC ||
				  stor_id == BLIS_CRC    ) KC = KC0;
		else if ( stor_id == BLIS_RCR ||
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4;
		else                               KC = KC0;
	}
	else /* if ( !packa && !packb ) */
	{
		if      ( stor_id == BLIS_RRR ||
				  stor_id == BLIS_CCC    ) KC = KC0;
		else if ( stor_id == BLIS_RRC ||
				  stor_id == BLIS_CRC    ) KC = KC0;
		else if ( stor_id == BLIS_RCR )
		{
		     if      ( m <=  4*MR ) KC = KC0;
		     else if ( m <= 36*MR ) KC = KC0 / 2;
		     else if ( m <= 56*MR ) KC = (( KC0 / 3 ) / 4 ) * 4;
		     else                   KC = KC0 / 4;
		}
		else if ( m <=   MR && n <=   NR ) KC = KC0;
		else if ( m <= 2*MR && n <= 2*NR ) KC = KC0 / 2;
		else if ( m <= 3*MR && n <= 3*NR ) KC = (( KC0 / 3 ) / 4 ) * 4;
		else if ( m <= 4*MR && n <= 4*NR ) KC = KC0 / 4;
		else                               KC = (( KC0 / 5 ) / 4 ) * 4;
	}

	/* Compute partitioning step values for each matrix of each loop. */
	const inc_t jcstep_c = cs_c;
	const inc_t jcstep_b = cs_b;

	const inc_t pcstep_a = cs_a;
	const inc_t pcstep_b = rs_b;

	const inc_t icstep_c = rs_c;
	const inc_t icstep_a = rs_a;

	const inc_t jrstep_c = cs_c * NR;

	const inc_t irstep_c = rs_c * MR;

	/*
	const inc_t jrstep_b = cs_b * NR;
	( void )jrstep_b;

	const inc_t irstep_c = rs_c * MR;
	const inc_t irstep_a = rs_a * MR;
	*/

	double ct[ BLIS_STACK_BUF_MAX_SIZE / sizeof( double ) ] __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE)));

	/* Storage scheme of ct should be same as that of C.
	   Since update routines only support row-major order,
	   col_pref flag is used to induce transpose to matrices before
	   passing to update routine whenever C is col-stored */
	const bool col_pref = (rs_c == 1) ? 1 : 0;

	const inc_t rs_ct = ( col_pref ? 1 : NR );
	const inc_t cs_ct = ( col_pref ? MR : 1 );

	double* restrict a_00       = a;
	double* restrict b_00       = b;
	double* restrict c_00       = c;
	double* restrict alpha_cast = alpha;
	double* restrict beta_cast  = beta;

	/* Make local copies of beta and one scalars to prevent any unnecessary
	   sharing of cache lines between the cores' caches. */
	double           beta_local = *beta_cast;
	double           one_local  = *PASTEMAC(d,1);

	auxinfo_t       aux;

	/* Parse and interpret the contents of the rntm_t object to properly
	   set the ways of parallelism for each loop. */
	/*bli_rntm_set_ways_from_rntm_sup( m, n, k, rntm );*/

	/* Initialize a mem_t entry for A and B. Strictly speaking, this is only
	   needed for the matrix we will be packing (if any), but we do it
	   unconditionally to be safe. An alternative way of initializing the
	   mem_t entries is:

	     bli_mem_clear( &mem_a );
	     bli_mem_clear( &mem_b );
	*/
	mem_t mem_a = BLIS_MEM_INITIALIZER;
	mem_t mem_b = BLIS_MEM_INITIALIZER;

	/* Define an array of bszid_t ids, which will act as our substitute for
	   the cntl_t tree. */
	/*                           5thloop  4thloop         packb  3rdloop         packa  2ndloop  1stloop  ukrloop */
	bszid_t bszids_nopack[6] = { BLIS_NC, BLIS_KC,               BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t bszids_packa [7] = { BLIS_NC, BLIS_KC,               BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t bszids_packb [7] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t bszids_packab[8] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t* restrict bszids;

	/* Set the bszids pointer to the correct bszids array above based on which
	   matrices (if any) are being packed. */
	if ( packa ) { if ( packb ) bszids = bszids_packab;
	               else         bszids = bszids_packa; }
	else         { if ( packb ) bszids = bszids_packb;
	               else         bszids = bszids_nopack; }

	/* Determine whether we are using more than one thread. */
	const bool is_mt = bli_rntm_calc_num_threads( rntm );

	thrinfo_t* restrict thread_jc = NULL;
	thrinfo_t* restrict thread_pc = NULL;
	thrinfo_t* restrict thread_pb = NULL;
	thrinfo_t* restrict thread_ic = NULL;
	thrinfo_t* restrict thread_pa = NULL;
	thrinfo_t* restrict thread_jr = NULL;

	/* Grow the thrinfo_t tree. */
	bszid_t*   restrict bszids_jc = bszids;
	                    thread_jc = thread;
	bli_thrinfo_sup_grow( rntm, bszids_jc, thread_jc );

	/* Compute the JC loop thread range for the current thread. */
	dim_t jc_start, jc_end;
	bli_thread_range_weighted_sub( thread_jc, 0, BLIS_UPPER, m, n, NR, FALSE, &jc_start, &jc_end );
	const dim_t n_local = jc_end - jc_start;

	dim_t m_off = 0;
	dim_t n_off = 0;
	doff_t diagoffc;
	dim_t m_off_cblock, n_off_cblock;
	dim_t jp, j;

	/* Compute number of primary and leftover components of the JC loop. */
	/*const dim_t jc_iter = ( n_local + NC - 1 ) / NC;*/
	const dim_t jc_left =   n_local % NC;

	/* Loop over the n dimension (NC rows/columns at a time). */
	/*for ( dim_t jj = 0; jj < jc_iter; jj += 1 )*/
	for ( dim_t jj = jc_start; jj < jc_end; jj += NC )
	{
		/* Calculate the thread's current JC block dimension. */
		const dim_t nc_cur = ( NC <= jc_end - jj ? NC : jc_left );

		double* restrict b_jc = b_00 + jj * jcstep_b;
		double* restrict c_jc = c_00 + jj * jcstep_c;

		/* Grow the thrinfo_t tree. */
		bszid_t*   restrict bszids_pc = &bszids_jc[1];
		                    thread_pc = bli_thrinfo_sub_node( thread_jc );
		bli_thrinfo_sup_grow( rntm, bszids_pc, thread_pc );

		/* Compute the PC loop thread range for the current thread. */
		const dim_t pc_start = 0, pc_end = k;
		const dim_t k_local = k;

		/* Compute number of primary and leftover components of the PC loop. */
		/*const dim_t pc_iter = ( k_local + KC - 1 ) / KC;*/
		const dim_t pc_left =   k_local % KC;

		/* Loop over the k dimension (KC rows/columns at a time). */
		/*for ( dim_t pp = 0; pp < pc_iter; pp += 1 )*/
		for ( dim_t pp = pc_start; pp < pc_end; pp += KC )
		{
			/* Calculate the thread's current PC block dimension. */
			const dim_t kc_cur = ( KC <= pc_end - pp ? KC : pc_left );

			double* restrict a_pc = a_00 + pp * pcstep_a;
			double* restrict b_pc = b_jc + pp * pcstep_b;

			/* Only apply beta to the first iteration of the pc loop. */
			double* restrict beta_use = ( pp == 0 ? &beta_local : &one_local );

			m_off = 0;
			n_off = jj;
			diagoffc = m_off - n_off;

			double* b_use;
			inc_t  rs_b_use, cs_b_use, ps_b_use;

			/* Set the bszid_t array and thrinfo_t pointer based on whether
			   we will be packing B. If we won't be packing B, we alias to
			   the _pc variables so that code further down can unconditionally
			   reference the _pb variables. Note that *if* we will be packing
			   B, the thrinfo_t node will have already been created by a
			   previous call to bli_thrinfo_grow(), since bszid values of
			   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
			   bszid that is a normal bszid_t value). */
			bszid_t*   restrict bszids_pb;
			if ( packb ) { bszids_pb = &bszids_pc[1];
			               thread_pb = bli_thrinfo_sub_node( thread_pc ); }
			else         { bszids_pb = &bszids_pc[0];
			               thread_pb = thread_pc; }

			/* Determine the packing buffer and related parameters for matrix
			   B. (If B will not be packed, then a_use will be set to point to
			   b and the _b_use strides will be set accordingly.) Then call
			   the packm sup variant chooser, which will call the appropriate
			   implementation based on the schema deduced from the stor_id. */
			PASTEMAC(d,packm_sup_b)
			(
			  packb,
			  BLIS_BUFFER_FOR_B_PANEL, /* This algorithm packs matrix B to */
			  stor_id,                 /* a "panel of B."                  */
			  BLIS_NO_TRANSPOSE,
			  KC,     NC,       /* This "panel of B" is (at most) KC x NC. */
			  kc_cur, nc_cur, NR,
			  &one_local,
			  b_pc,   rs_b,      cs_b,
			  &b_use, &rs_b_use, &cs_b_use,
			                     &ps_b_use,
			  cntx,
			  rntm,
			  &mem_b,
			  thread_pb 
			);

			/* Alias a_use so that it's clear this is our current block of
			   matrix B. */
			double* restrict b_pc_use = b_use;

			/* We don't need to embed the panel stride of B within the auxinfo_t
			   object because this variant iterates through B in the jr loop,
			   which occurs here, within the macrokernel, not within the
			   millikernel. */
			/*bli_auxinfo_set_ps_b( ps_b_use, &aux );*/

			/* Grow the thrinfo_t tree. */
			bszid_t*   restrict bszids_ic = &bszids_pb[1];
			                    thread_ic = bli_thrinfo_sub_node( thread_pb );
			bli_thrinfo_sup_grow( rntm, bszids_ic, thread_ic );

			/* Compute the IC loop thread range for the current thread. */
			dim_t ic_start, ic_end;
			bli_thread_range_weighted_sub( thread_ic, -diagoffc, BLIS_LOWER, nc_cur, m, MR, FALSE, &ic_start, &ic_end );
			const dim_t m_local = ic_end - ic_start;

			/* Compute number of primary and leftover components of the IC loop. */
			/*const dim_t ic_iter = ( m_local + MC - 1 ) / MC;*/
			const dim_t ic_left =   m_local % MC;

			/* Loop over the m dimension (MC rows at a time). */
			/*for ( dim_t ii = 0; ii < ic_iter; ii += 1 )*/
			for ( dim_t ii = ic_start; ii < ic_end; ii += MC )
			{
				/* Calculate the thread's current IC block dimension. */
				dim_t mc_cur = ( MC <= ic_end - ii ? MC : ic_left );

				dim_t nc_pruned = nc_cur;

				m_off = ii;
				n_off = jj;

				if(bli_gemmt_is_strictly_below_diag(m_off, n_off, mc_cur, nc_cur)) continue;

				double* restrict a_ic = a_pc + ii * icstep_a;
				double* restrict c_ic = c_jc + ii * icstep_c;

				doff_t diagoffc = m_off - n_off;

				double* restrict b_pc_pruned = b_pc_use;

				if(diagoffc > 0 )
				{
					jp = diagoffc / NR;
					j = jp * NR;
					nc_pruned = nc_cur - j;
					n_off += j;
					diagoffc = diagoffc % NR;
					c_ic = c_ic + ( j ) * cs_c;
					b_pc_pruned = b_pc_use + ( jp ) * ps_b_use;
				}

				if( ( ( -diagoffc ) + nc_pruned ) < mc_cur )
				{
					mc_cur = -diagoffc + nc_pruned;
				}

				double* a_use;
				inc_t  rs_a_use, cs_a_use, ps_a_use;

				/* Set the bszid_t array and thrinfo_t pointer based on whether
				   we will be packing B. If we won't be packing A, we alias to
				   the _ic variables so that code further down can unconditionally
				   reference the _pa variables. Note that *if* we will be packing
				   A, the thrinfo_t node will have already been created by a
				   previous call to bli_thrinfo_grow(), since bszid values of
				   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
				   bszid that is a normal bszid_t value). */
				bszid_t*   restrict bszids_pa;
				if ( packa ) { bszids_pa = &bszids_ic[1];
							   thread_pa = bli_thrinfo_sub_node( thread_ic ); }
				else         { bszids_pa = &bszids_ic[0];
							   thread_pa = thread_ic; }

				/* Determine the packing buffer and related parameters for matrix
				   A. (If A will not be packed, then a_use will be set to point to
				   a and the _a_use strides will be set accordingly.) Then call
				   the packm sup variant chooser, which will call the appropriate
				   implementation based on the schema deduced from the stor_id. */
				PASTEMAC(d,packm_sup_a)
				(
				  packa,
				  BLIS_BUFFER_FOR_A_BLOCK, /* This algorithm packs matrix A to */
				  stor_id,                 /* a "block of A."                  */
				  BLIS_NO_TRANSPOSE,
				  MC,     KC,       /* This "block of A" is (at most) MC x KC. */
				  mc_cur, kc_cur, MR,
				  &one_local,
				  a_ic,   rs_a,      cs_a,
				  &a_use, &rs_a_use, &cs_a_use,
				                     &ps_a_use,
				  cntx,
				  rntm,
				  &mem_a,
				  thread_pa 
				);

				/* Alias a_use so that it's clear this is our current block of
				   matrix A. */
				double* restrict a_ic_use = a_use;

				/* Embed the panel stride of A within the auxinfo_t object. The
				   millikernel will query and use this to iterate through
				   micropanels of A (if needed). */
				bli_auxinfo_set_ps_a( ps_a_use, &aux );

				/* Grow the thrinfo_t tree. */
				bszid_t*   restrict bszids_jr = &bszids_pa[1];
				                    thread_jr = bli_thrinfo_sub_node( thread_pa );
				bli_thrinfo_sup_grow( rntm, bszids_jr, thread_jr );

				/* Compute number of primary and leftover components of the JR loop. */
				dim_t jr_iter = ( nc_pruned + NR - 1 ) / NR;
				dim_t jr_left =   nc_pruned % NR;

				/* Compute the JR loop thread range for the current thread. */
				dim_t jr_start, jr_end;
				bli_thread_range_sub( thread_jr, jr_iter, 1, FALSE, &jr_start, &jr_end );

				/* An optimization: allow the last jr iteration to contain up to NRE
				   columns of C and B. (If NRE > NR, the mkernel has agreed to handle
				   these cases.) Note that this prevents us from declaring jr_iter and
				   jr_left as const. NOTE: We forgo this optimization when packing B
				   since packing an extended edge case is not yet supported. */
				if ( !packb && !is_mt )
				if ( NRE != 0 && 1 < jr_iter && jr_left != 0 && jr_left <= NRE )
				{
					jr_iter--; jr_left += NR;
				}

				/* Loop over the n dimension (NR columns at a time). */
				/*for ( dim_t j = 0; j < jr_iter; j += 1 )*/
				for ( dim_t j = jr_start; j < jr_end; j += 1 )
				{
					const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? NR : jr_left );

					/*
					double* restrict b_jr = b_pc_use + j * jrstep_b;
					*/
					double* restrict b_jr = b_pc_pruned + j * ps_b_use;
					double* restrict c_jr = c_ic     + j * jrstep_c;
					dim_t m_rect = 0;
				        dim_t n_iter_rect = 0;

					m_off_cblock = m_off;
					n_off_cblock = n_off + j * NR;

					if(bli_gemmt_is_strictly_above_diag(m_off_cblock, n_off_cblock, mc_cur, nr_cur))
					{
						m_rect = mc_cur;
					}
					else
					{
						/* calculate the number of rows in rectangular region of the block */
						n_iter_rect = n_off_cblock < m_off_cblock ? 0: (n_off_cblock - m_off_cblock) / MR;
						m_rect = n_iter_rect * MR;
					}

					/* Compute the rectangular part */
					gemmsup_ker
					(
					  conja,
					  conjb,
					  m_rect,
					  nr_cur,
					  kc_cur,
					  alpha_cast,
					  a_ic_use, rs_a_use, cs_a_use,
					  b_jr,     rs_b_use, cs_b_use,
					  beta_use,
					  c_jr,     rs_c,     cs_c,
					  &aux,
					  cntx 
					);

					m_off_cblock = m_off + m_rect;

					double* restrict a_ir = a_ic_use + n_iter_rect * ps_a_use;
					double* restrict c_ir = c_jr + n_iter_rect * irstep_c;

					/* compute the remaining triangular part */
					for( dim_t i = m_rect;( i < mc_cur) && (m_off_cblock < n_off_cblock + nr_cur); i += MR )
					{
						const dim_t mr_cur = (i+MR-1) < mc_cur ? MR : mc_cur - i;
						dim_t m_off_24 = m_off_cblock % 24;
						dim_t n_off_24 = n_off_cblock % 24;
						dim_t m_idx = (dim_t)(m_off_24 / MR);
						dim_t n_idx = (dim_t)(n_off_24 / NR);
					#ifdef BLIS_KERNELS_ZEN4
						if ( (n_idx == m_idx) && (MR == 24) && (NR == 8) && bli_cpuid_is_avx512_supported() &&
								(stor_id != BLIS_CRC && stor_id != BLIS_RRC) &&
								// verify if micro panel intersects with diagonal
								// if distance from diagonal (n_off_cblock - m_off_cblock) is greater
								// than (LCM(MR, NR) - NR) then it implies that micro panel is far
								// from diagonal therefore it it does not intersect with it.
								(n_off_cblock - m_off_cblock) <= 16 // (n_off_cblock - m_off_cblock) <= (LCM(MR, NR) - NR)
							)
						{
							/*
								call traingular 24x8 DGEMMT kernels
							*/
							// Difference between n_off_cblock and m_off_cblock is same as
							// the size of full GEMM region.
							// kernel_idx = 0 is used when full GEMM region size <= 0
							// kernel_idx = 1 is used when full GEMM region size <= 8
							// kernel_idx = 2 is used when full GEMM region size <= 16
							ker_fpus_zen4[(n_off_cblock - m_off_cblock)/NR]
							(
								conja,
								conjb,
								mr_cur,
								nr_cur,
								kc_cur,
								(double *)alpha_cast,
								(double *)a_ir, rs_a_use, cs_a_use,
								(double *)b_jr, rs_b_use, cs_b_use,
								(double *)beta_use,
								(double *)c_ir, rs_c, cs_c,
								&aux,
								cntx
							);
							a_ir += ps_a_use;
							c_ir += irstep_c;
							m_off_cblock += mr_cur;
							continue;
						}
					#endif
					#ifdef BLIS_KERNELS_HASWELL
						/* Prerequisites : MR = 6, NR = 8.
						An optimization: allow the last jr iteration to contain up to NRE
						In DGEMMT API implementation, kernel operates on 6x8 block. MR and
						NR are set as 6 and 8 respectively. 24 being the LCM of 6 and 8,
						the diagonal pattern repeats for every 24x24 block.
						This pattern is exploited to achieve the optimization in diagonal
						blocks by computing only the required elements. In the previous
						implementation, all the 48 outputs of the given 6x8 block are
						computed and stored into a temporary buffer. Later, the required
						elements are copied into the final C output buffer.
						With this optimization, we are avoiding copy operation and also
						reducing the number of computations.
						Variables m_off_24 and n_off_24 respectively store the m and n
						offsets from the starting point of the corresponding 24x24 block.
						Variables m_idx and n_idx store indices of the current 6x8 block
						along m and n dimensions, in 24x24 block. m_idx is computed as
						(m_off_24 / MR) while n_idx is computed as (n_off_24 / NR).
						Range of m_idx is 0 <= m_idx <= 3 and the range of n_idx is
						0 <= n_idx <= 2. Based on these indices, for the given 6x8 block,
						logic is implemented to identify the relevant kernel from the
						look-up table.
						During instances, where m is not a multiple of 6 or n is not a
						multiple of 8, it goes to the default gemm kernel. MR and NR must be
						6 and 8 for these kernels to achieve the expected functionality.*/
						// dim_t m_off_24 = m_off_cblock % 24;
						// dim_t n_off_24 = n_off_cblock % 24;
						// dim_t m_idx = (dim_t)(m_off_24 / MR);
						// dim_t n_idx = (dim_t)(n_off_24 / NR);

						/* Check if m, n indices are multiple of MR and NR respectively
							and current block is a complete 6x8 block */
						bool idx_supported = ((m_off_24 % MR) == 0) && ((n_off_24 % NR) == 0)
						&& (MR == 6) && (NR == 8)
						&& (bli_cpuid_is_avx2fma3_supported() == TRUE) && (mr_cur==MR) && (nr_cur==NR);

						/* m_idx and n_idx would be equal only if the current block is
							a diagonal block */
						if( (dt == BLIS_DOUBLE) && (m_idx == n_idx) && idx_supported )
						{
							dim_t ker_idx = m_idx<<1;
							/* If there is another 6x8 diagonal block pending for computation
								after the current 6x8 diagonal block, then the two blocks can
								be computed together(12x8). This combined kernel is implemented
								only for the case where n_idx = 0 i.e., n_off_24 = 0. To call
								this, it has to be ensured that at least 12 rows are pending in
								C for computation (i+ MR + MR <= mc_cur). Usage of this combined
								kernel saves the entire time to execute one kernel*/
							if( (n_idx == 0) && (i+ MR + MR <= mc_cur) ) {
								ker_idx = 6; /* use combined kernel, index of combined kernel
												in lookup table is 6 */
							}
							/* if B is column storage we use rd kernel*/
							if( stor_id == BLIS_RRC ) {
								ker_idx += 7; /* index of rd kernel*/
							}
							gemmt_ker_ft ker_fp = ker_fpus_haswell[ker_idx];
							ker_fp
							(
								conja,
								conjb,
								mr_cur,
								nr_cur,
								kc_cur,
								(double*) alpha_cast,
								(double*) a_ir, rs_a_use, cs_a_use,
								(double*) b_jr,     rs_b_use, cs_b_use,
								(double*) beta_use,
								(double*) c_ir, rs_c, cs_c, 
								&aux,
								cntx 
							);
							a_ir += ps_a_use;
							c_ir += irstep_c;
							m_off_cblock += mr_cur;
							continue;
						}
						/* 6x8 block where m_idx == n_idx+1 also has some parts of the diagonal */
						else if ( (dt == BLIS_DOUBLE) && (m_idx == n_idx+1) && (idx_supported) ) 
						{
							/* If current block was already computed in the combined kernel it
								can be skipped combined kernel is only implemented for n_idx=0,
								i == m_rect is only true for the first iteration therefore if
								i == m_rect then the current 6x8 block was not computed in
								combined kernel
							*/
							if ( (n_idx != 0) || (i == m_rect) )
							{
								dim_t ker_idx = (n_idx << 1) + 1 ;
								/* use rd kernel if B is column major storage */
								if( stor_id == BLIS_RRC ) { ker_idx += 7; }
								
								gemmt_ker_ft ker_fp = ker_fpus_haswell[ker_idx];
								
								ker_fp
								(
									conja,
									conjb,
									mr_cur,
									nr_cur,
									kc_cur,
									(double*) alpha_cast,
									(double*) a_ir, rs_a_use, cs_a_use,
									(double*) b_jr,     rs_b_use, cs_b_use,
									(double*) beta_use,
									(double*) c_ir,     rs_c,     cs_c,
									&aux,
									cntx 
								);
							}
							a_ir += ps_a_use;
							c_ir += irstep_c;
							m_off_cblock += mr_cur;
							continue;
						}
					#endif
						gemmsup_ker
						(
							conja,
							conjb,
							mr_cur,
							nr_cur,
							kc_cur,
							alpha_cast,
							a_ir, rs_a_use, cs_a_use,
							b_jr,     rs_b_use, cs_b_use,
							zero,
							ct,     rs_ct,     cs_ct, 
							&aux,
							cntx 
						);

						if( col_pref )
						{
							PASTEMAC(d,update_lower_triang)( n_off_cblock, m_off_cblock, 
								nr_cur, mr_cur,
								ct, cs_ct, rs_ct,
								beta_use,
								c_ir, cs_c, rs_c );
						}
						else
						{
							PASTEMAC(d,update_upper_triang)( m_off_cblock, n_off_cblock, 
								mr_cur, nr_cur,
								ct, rs_ct, cs_ct,
								beta_use,
								c_ir, rs_c, cs_c );
						}

						a_ir += ps_a_use;
						c_ir += irstep_c;
						m_off_cblock += mr_cur;

					}
				}
			}

			/* NOTE: This barrier is only needed if we are packing B (since
			   that matrix is packed within the pc loop of this variant). */
			if ( packb ) bli_thread_barrier( thread_pb );
		}
	}

	/* Release any memory that was acquired for packing matrices A and B. */
	PASTEMAC(d,packm_sup_finalize_mem_a)
	(
	  packa,
	  rntm,
	  &mem_a,
	  thread_pa 
	);
	PASTEMAC(d,packm_sup_finalize_mem_b)
	(
	  packb,
	  rntm,
	  &mem_b,
	  thread_pb 
	);

/*
PASTEMAC(d,fprintm)( stdout, "gemmsup_ref_var2: b1", kc_cur, nr_cur, b_jr, rs_b, cs_b, "%4.1f", "" );
PASTEMAC(d,fprintm)( stdout, "gemmsup_ref_var2: a1", mr_cur, kc_cur, a_ir, rs_a, cs_a, "%4.1f", "" );
PASTEMAC(d,fprintm)( stdout, "gemmsup_ref_var2: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%4.1f", "" );
*/
}

/***************************************************************/
/* AVX512 Kernel - gemmsup_rv_zen4_asm_4x4m                    */
/* Check if AVX512 kernel can be called for certain conditions */
/* 1. Architecture: ZEN4 or ZEN5                               */
/* 2. Storage: If it is CRC, RRC AVX2 code path is invoked     */
/*              for other storage formats AVX512 will be called*/
/* 3. BlockSize: Kernel is optimised for MR=NR=4               */
/***************************************************************/
#if defined (BLIS_KERNELS_ZEN4)

#define LOWER_TRIANGLE_OPTIMIZATION_DCOMPLEX() \
	if ((MR == 4) && (NR == 4) && (stor_id != BLIS_CRC) && (stor_id != BLIS_RRC)) \
	{ \
		bli_zgemmsup_rv_zen4_asm_4x4m_lower \
			( \
				conja, \
				conjb, \
				mr_cur, \
				nr_cur, \
				kc_cur, \
				(dcomplex*) alpha_cast, \
				(dcomplex*) a_ir, rs_a_use, cs_a_use, \
				(dcomplex*) b_jr,     rs_b_use, cs_b_use, \
				(dcomplex*) beta_use, \
				(dcomplex*) c_ir,     rs_c,     cs_c, \
				&aux, \
				cntx \
			); \
	} \
	/* call the regular kernel for non applicable cases */ \
	else

#define UPPER_TRIANGLE_OPTIMIZATION_DCOMPLEX() \
	if ((MR == 4) && (NR == 4) && (stor_id != BLIS_CRC) && (stor_id != BLIS_RRC)) \
		{ \
			bli_zgemmsup_rv_zen4_asm_4x4m_upper \
				( \
					conja, \
					conjb, \
					mr_cur, \
					nr_cur, \
					kc_cur, \
					(dcomplex*) alpha_cast, \
					(dcomplex*) a_ir, rs_a_use, cs_a_use, \
					(dcomplex*) b_jr,     rs_b_use, cs_b_use, \
					(dcomplex*) beta_use, \
					(dcomplex*) c_ir,     rs_c,     cs_c, \
					&aux, \
					cntx \
				); \
		} \
		/* call the regular kernel for non applicable cases */ \
		else

#else
	#define LOWER_TRIANGLE_OPTIMIZATION_DCOMPLEX()
	#define UPPER_TRIANGLE_OPTIMIZATION_DCOMPLEX()

#endif

void bli_zgemmtsup_l_ref_var2m
     ( \
       bool             packa,
       bool             packb,
       conj_t           conja,
       conj_t           conjb,
       dim_t            m,
       dim_t            n,
       dim_t            k,
       void*   restrict alpha,
       void*   restrict a, inc_t rs_a, inc_t cs_a,
       void*   restrict b, inc_t rs_b, inc_t cs_b,
       void*   restrict beta,
       void*   restrict c, inc_t rs_c, inc_t cs_c,
       stor3_t          stor_id,
       cntx_t* restrict cntx,
       rntm_t* restrict rntm,
       thrinfo_t* restrict thread 
     )
{
	const num_t dt = PASTEMAC(z,type);

	dcomplex* restrict zero = PASTEMAC(z,0);

	/* If m or n is zero, return immediately. */
	if ( bli_zero_dim2( m, n ) ) return;

	/* If k < 1 or alpha is zero, scale by beta and return. */
	if ( k < 1 || PASTEMAC(z,eq0)( *(( dcomplex* )alpha) ) )
	{
		if ( bli_thread_am_ochief( thread ) )
		{
			PASTEMAC(z,scalm)
			(
			  BLIS_NO_CONJUGATE,
			  0,
			  BLIS_NONUNIT_DIAG,
			  BLIS_DENSE,
			  m, n,
			  beta,
			  c, rs_c, cs_c
			);
		}
		return;
	}

	/* Query the context for various blocksizes. */
	dim_t NR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NR, cntx );
	dim_t MR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MR, cntx );
	dim_t NC  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NC, cntx );
	dim_t MC  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MC, cntx );
	dim_t KC0 = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_KC, cntx );
	/* Query the maximum blocksize for NR, which implies a maximum blocksize
	   extension for the final iteration. */
	dim_t NRM = bli_cntx_get_l3_sup_tri_blksz_max_dt( dt, BLIS_NR, cntx );

	/* Query the context for the sup microkernel address and cast it to its
	   function pointer type. */
	PASTECH(z,gemmsup_ker_ft)
               gemmsup_ker = bli_cntx_get_l3_sup_tri_ker_dt( dt, stor_id, cntx );

	if( ( 0 == NR ) || ( 0 == MR ) || ( 0 == NC ) || ( 0 == MC ) || ( 0 == KC0 ) )
	{
		NR = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx );
		MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx );
		NC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx );
		MC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx );
		KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx );
		NRM = bli_cntx_get_l3_sup_blksz_max_dt( dt, BLIS_NR, cntx );
		gemmsup_ker = bli_cntx_get_l3_sup_ker_dt( dt, stor_id, cntx );
	}
	const dim_t NRE = NRM - NR;

	dim_t KC;
	if      ( packa && packb )
	{
		KC = KC0;
	}
	else if ( packb )
	{
		if      ( stor_id == BLIS_RRR ||
				  stor_id == BLIS_CCC    ) KC = KC0;
		else if ( stor_id == BLIS_RRC ||
				  stor_id == BLIS_CRC    ) KC = KC0;
		else if ( stor_id == BLIS_RCR ||
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4;
		else                               KC = KC0;
	}
	else if ( packa )
	{
		if      ( stor_id == BLIS_RRR ||
				  stor_id == BLIS_CCC    ) KC = (( KC0 / 2 ) / 2 ) * 2;
		else if ( stor_id == BLIS_RRC ||
				  stor_id == BLIS_CRC    ) KC = KC0;
		else if ( stor_id == BLIS_RCR ||
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4;
		else                               KC = KC0;
	}
	else /* if ( !packa && !packb ) */
	{
		if      ( stor_id == BLIS_RRR ||
				  stor_id == BLIS_CCC    ) KC = KC0;
		else if ( stor_id == BLIS_RRC ||
				  stor_id == BLIS_CRC    ) KC = KC0;
		else if ( m <=   MR && n <=   NR ) KC = KC0;
		else if ( m <= 2*MR && n <= 2*NR ) KC = KC0 / 2;
		else if ( m <= 3*MR && n <= 3*NR ) KC = (( KC0 / 3 ) / 4 ) * 4;
		else if ( m <= 4*MR && n <= 4*NR ) KC = KC0 / 4;
		else                               KC = (( KC0 / 5 ) / 4 ) * 4;
	}

	/* Compute partitioning step values for each matrix of each loop. */
	const inc_t jcstep_c = cs_c;
	const inc_t jcstep_b = cs_b;

	const inc_t pcstep_a = cs_a;
	const inc_t pcstep_b = rs_b;

	const inc_t icstep_c = rs_c;
	const inc_t icstep_a = rs_a;

	const inc_t jrstep_c = cs_c * NR;

	const inc_t irstep_c = rs_c * MR;

	/*
	const inc_t jrstep_b = cs_b * NR;
	( void )jrstep_b;

	const inc_t irstep_c = rs_c * MR;
	const inc_t irstep_a = rs_a * MR;
	*/

	dcomplex ct[ BLIS_STACK_BUF_MAX_SIZE / sizeof( dcomplex ) ]  __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE)));

	/* storage-scheme of ct should be same as that of C.
	  Since update routines only support row-major order,
	  col_pref flag is used to induce transpose to matrices before
	  passing to update routine whenever C is col-stored */
	const bool col_pref = (rs_c == 1)? 1 : 0;

	const inc_t rs_ct = ( col_pref ? 1 : NR );
	const inc_t cs_ct = ( col_pref ? MR : 1 );

	dcomplex* restrict a_00       = a;
	dcomplex* restrict b_00       = b;
	dcomplex* restrict c_00       = c;
	dcomplex* restrict alpha_cast = alpha;
	dcomplex* restrict beta_cast  = beta;

	/* Make local copies of beta and one scalars to prevent any unnecessary
	   sharing of cache lines between the cores' caches. */ \
	dcomplex           beta_local = *beta_cast;
	dcomplex           one_local  = *PASTEMAC(z,1);

	auxinfo_t       aux;

	/* Parse and interpret the contents of the rntm_t object to properly
	   set the ways of parallelism for each loop. */ 
	/*bli_rntm_set_ways_from_rntm_sup( m, n, k, rntm );*/

	/* Initialize a mem_t entry for A and B. Strictly speaking, this is only
	   needed for the matrix we will be packing (if any), but we do it
	   unconditionally to be safe. An alternative way of initializing the
	   mem_t entries is:

	     bli_mem_clear( &mem_a );
	     bli_mem_clear( &mem_b );
	*/
	mem_t mem_a = BLIS_MEM_INITIALIZER;
	mem_t mem_b = BLIS_MEM_INITIALIZER;

	/* Define an array of bszid_t ids, which will act as our substitute for
	   the cntl_t tree. */
	/*                           5thloop  4thloop         packb  3rdloop         packa  2ndloop  1stloop  ukrloop */
	bszid_t bszids_nopack[6] = { BLIS_NC, BLIS_KC,               BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t bszids_packa [7] = { BLIS_NC, BLIS_KC,               BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t bszids_packb [7] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t bszids_packab[8] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t* restrict bszids;

	/* Set the bszids pointer to the correct bszids array above based on which
	   matrices (if any) are being packed. */
	if ( packa ) { if ( packb ) bszids = bszids_packab;
	               else         bszids = bszids_packa; }
	else         { if ( packb ) bszids = bszids_packb;
	               else         bszids = bszids_nopack; }

	/* Determine whether we are using more than one thread. */
	const bool is_mt = bli_rntm_calc_num_threads( rntm );

	thrinfo_t* restrict thread_jc = NULL;
	thrinfo_t* restrict thread_pc = NULL;
	thrinfo_t* restrict thread_pb = NULL;
	thrinfo_t* restrict thread_ic = NULL;
	thrinfo_t* restrict thread_pa = NULL;
	thrinfo_t* restrict thread_jr = NULL;

	/* Grow the thrinfo_t tree. */
	bszid_t*   restrict bszids_jc = bszids;
	                    thread_jc = thread;
	bli_thrinfo_sup_grow( rntm, bszids_jc, thread_jc );

	/* Compute the JC loop thread range for the current thread. */
	dim_t jc_start, jc_end;
	bli_thread_range_weighted_sub( thread_jc, 0, BLIS_LOWER, m, n, NR, FALSE, &jc_start, &jc_end );
	const dim_t n_local = jc_end - jc_start;

	/* Compute number of primary and leftover components of the JC loop. */
	/*const dim_t jc_iter = ( n_local + NC - 1 ) / NC;*/
	const dim_t jc_left =   n_local % NC;

	dim_t m_off_cblock, n_off_cblock;
	dim_t m_off = 0;
	dim_t n_off = 0;
	doff_t diagoffc;
	dim_t i, ip;

	/* Loop over the n dimension (NC rows/columns at a time). */
	/*for ( dim_t jj = 0; jj < jc_iter; jj += 1 )*/
	for ( dim_t jj = jc_start; jj < jc_end; jj += NC )
	{
		/* Calculate the thread's current JC block dimension. */
		const dim_t nc_cur = ( NC <= jc_end - jj ? NC : jc_left );

		dcomplex* restrict b_jc = b_00 + jj * jcstep_b;
		dcomplex* restrict c_jc = c_00 + jj * jcstep_c;

		/* Grow the thrinfo_t tree. */
		bszid_t*   restrict bszids_pc = &bszids_jc[1];
		                    thread_pc = bli_thrinfo_sub_node( thread_jc );
		bli_thrinfo_sup_grow( rntm, bszids_pc, thread_pc );

		/* Compute the PC loop thread range for the current thread. */
		const dim_t pc_start = 0, pc_end = k;
		const dim_t k_local = k;

		/* Compute number of primary and leftover components of the PC loop. */
		/*const dim_t pc_iter = ( k_local + KC - 1 ) / KC;*/
		const dim_t pc_left =   k_local % KC;

		/* Loop over the k dimension (KC rows/columns at a time). */
		/*for ( dim_t pp = 0; pp < pc_iter; pp += 1 )*/
		for ( dim_t pp = pc_start; pp < pc_end; pp += KC )
		{
			/* Calculate the thread's current PC block dimension. */
			const dim_t kc_cur = ( KC <= pc_end - pp ? KC : pc_left );

			dcomplex* restrict a_pc = a_00 + pp * pcstep_a;
			dcomplex* restrict b_pc = b_jc + pp * pcstep_b;

			/* Only apply beta to the first iteration of the pc loop. */
			dcomplex* restrict beta_use = ( pp == 0 ? &beta_local : &one_local );

			m_off = 0;
			n_off = jj;
			diagoffc = m_off - n_off;

			dcomplex* b_use;
			inc_t  rs_b_use, cs_b_use, ps_b_use;

			/* Set the bszid_t array and thrinfo_t pointer based on whether
			   we will be packing B. If we won't be packing B, we alias to
			   the _pc variables so that code further down can unconditionally
			   reference the _pb variables. Note that *if* we will be packing
			   B, the thrinfo_t node will have already been created by a
			   previous call to bli_thrinfo_grow(), since bszid values of
			   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
			   bszid that is a normal bszid_t value). */
			bszid_t*   restrict bszids_pb;
			if ( packb ) { bszids_pb = &bszids_pc[1];
			               thread_pb = bli_thrinfo_sub_node( thread_pc ); }
			else         { bszids_pb = &bszids_pc[0];
			               thread_pb = thread_pc; }

			/* Determine the packing buffer and related parameters for matrix
			   B. (If B will not be packed, then a_use will be set to point to
			   b and the _b_use strides will be set accordingly.) Then call
			   the packm sup variant chooser, which will call the appropriate
			   implementation based on the schema deduced from the stor_id. */ \
			PASTEMAC(z,packm_sup_b)
			(
			  packb,
			  BLIS_BUFFER_FOR_B_PANEL, /* This algorithm packs matrix B to */
			  stor_id,                 /* a "panel of B."                  */
			  BLIS_NO_TRANSPOSE,
			  KC,     NC,       /* This "panel of B" is (at most) KC x NC. */
			  kc_cur, nc_cur, NR,
			  &one_local,
			  b_pc,   rs_b,      cs_b,
			  &b_use, &rs_b_use, &cs_b_use,
			                     &ps_b_use,
			  cntx,
			  rntm,
			  &mem_b,
			  thread_pb 
			);

			/* Alias a_use so that it's clear this is our current block of
			   matrix B. */
			dcomplex* restrict b_pc_use = b_use;

			/* We don't need to embed the panel stride of B within the auxinfo_t
			   object because this variant iterates through B in the jr loop,
			   which occurs here, within the macrokernel, not within the
			   millikernel. */
			/*bli_auxinfo_set_ps_b( ps_b_use, &aux );*/

			/* Grow the thrinfo_t tree. */
			bszid_t*   restrict bszids_ic = &bszids_pb[1];
			                    thread_ic = bli_thrinfo_sub_node( thread_pb );
			bli_thrinfo_sup_grow( rntm, bszids_ic, thread_ic );

			/* Compute the IC loop thread range for the current thread. */
			dim_t ic_start, ic_end;
			bli_thread_range_weighted_sub( thread_ic, -diagoffc, BLIS_UPPER, nc_cur, m, MR, FALSE, &ic_start, &ic_end );
			const dim_t m_local = ic_end - ic_start;

			/* Compute number of primary and leftover components of the IC loop. */
			/*const dim_t ic_iter = ( m_local + MC - 1 ) / MC;*/
			const dim_t ic_left =   m_local % MC;

			/* Loop over the m dimension (MC rows at a time). */
			/*for ( dim_t ii = 0; ii < ic_iter; ii += 1 )*/
			for ( dim_t ii = ic_start; ii < ic_end; ii += MC )
			{
				/* Calculate the thread's current IC block dimension. */
				dim_t mc_cur = ( MC <= ic_end - ii ? MC : ic_left );
				dim_t nc_pruned = nc_cur;

				dcomplex* restrict a_ic = a_pc + ii * icstep_a;
				dcomplex* restrict c_ic = c_jc + ii * icstep_c;

				m_off = ii;

				if(bli_gemmt_is_strictly_above_diag( m_off, n_off, mc_cur, nc_cur ) ) continue;

				diagoffc = m_off - n_off;

				if( diagoffc < 0 )
				{
					ip = -diagoffc / MR;
					i = ip * MR;
					mc_cur = mc_cur - i;
					diagoffc = -diagoffc % MR;
					m_off += i;
					c_ic = c_ic + ( i ) * rs_c;
					a_ic = a_ic + ( i ) * rs_a;
				}

				if( ( diagoffc + mc_cur ) < nc_cur )
				{
					nc_pruned = diagoffc + mc_cur;
				}

				dcomplex* a_use;
				inc_t  rs_a_use, cs_a_use, ps_a_use;

				/* Set the bszid_t array and thrinfo_t pointer based on whether
				   we will be packing B. If we won't be packing A, we alias to
				   the _ic variables so that code further down can unconditionally
				   reference the _pa variables. Note that *if* we will be packing
				   A, the thrinfo_t node will have already been created by a
				   previous call to bli_thrinfo_grow(), since bszid values of
				   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
				   bszid that is a normal bszid_t value). */
				bszid_t*   restrict bszids_pa;
				if ( packa ) { bszids_pa = &bszids_ic[1];
							   thread_pa = bli_thrinfo_sub_node( thread_ic ); }
				else         { bszids_pa = &bszids_ic[0];
							   thread_pa = thread_ic; }

				/* Determine the packing buffer and related parameters for matrix
				   A. (If A will not be packed, then a_use will be set to point to
				   a and the _a_use strides will be set accordingly.) Then call
				   the packm sup variant chooser, which will call the appropriate
				   implementation based on the schema deduced from the stor_id. */ \
				PASTEMAC(z,packm_sup_a)
				(
				  packa,
				  BLIS_BUFFER_FOR_A_BLOCK, /* This algorithm packs matrix A to */
				  stor_id,                 /* a "block of A."                  */
				  BLIS_NO_TRANSPOSE,
				  MC,     KC,       /* This "block of A" is (at most) MC x KC. */
				  mc_cur, kc_cur, MR,
				  &one_local,
				  a_ic,   rs_a,      cs_a,
				  &a_use, &rs_a_use, &cs_a_use,
				                     &ps_a_use,
				  cntx,
				  rntm,
				  &mem_a,
				  thread_pa 
				);

				/* Alias a_use so that it's clear this is our current block of
				   matrix A. */
				dcomplex* restrict a_ic_use = a_use;

				/* Embed the panel stride of A within the auxinfo_t object. The
				   millikernel will query and use this to iterate through
				   micropanels of A (if needed). */
				bli_auxinfo_set_ps_a( ps_a_use, &aux );

				/* Grow the thrinfo_t tree. */
				bszid_t*   restrict bszids_jr = &bszids_pa[1];
				                    thread_jr = bli_thrinfo_sub_node( thread_pa );
				bli_thrinfo_sup_grow( rntm, bszids_jr, thread_jr );

				/* Compute number of primary and leftover components of the JR loop. */
				dim_t jr_iter = ( nc_pruned + NR - 1 ) / NR;
				dim_t jr_left =   nc_pruned % NR;

				/* Compute the JR loop thread range for the current thread. */
				dim_t jr_start, jr_end;
				bli_thread_range_sub( thread_jr, jr_iter, 1, FALSE, &jr_start, &jr_end );

				/* An optimization: allow the last jr iteration to contain up to NRE
				   columns of C and B. (If NRE > NR, the mkernel has agreed to handle
				   these cases.) Note that this prevents us from declaring jr_iter and
				   jr_left as const. NOTE: We forgo this optimization when packing B
				   since packing an extended edge case is not yet supported. */
				if ( !packb && !is_mt )
				if ( NRE != 0 && 1 < jr_iter && jr_left != 0 && jr_left <= NRE )
				{
					jr_iter--; jr_left += NR;
				}

				/* Loop over the n dimension (NR columns at a time). */
				/*for ( dim_t j = 0; j < jr_iter; j += 1 )*/
				for ( dim_t j = jr_start; j < jr_end; j += 1 )
				{
					const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? NR : jr_left );

					/*
					dcomplex* restrict b_jr = b_pc_use + j * jrstep_b;
					*/
					dcomplex* restrict b_jr = b_pc_use + j * ps_b_use;
					dcomplex* restrict c_jr = c_ic     + j * jrstep_c;

					dim_t i;
					dim_t m_zero = 0;
					dim_t n_iter_zero = 0;

					m_off_cblock = m_off;
					n_off_cblock = n_off + j * NR;

					if(bli_gemmt_is_strictly_below_diag(m_off_cblock, n_off_cblock, mc_cur, nc_cur))
					{
						m_zero = 0;
					}
					else
					{
						/* compute number of rows that are filled with zeroes and can be ignored */
						n_iter_zero = (n_off_cblock < m_off_cblock)? 0 : (n_off_cblock - m_off)/MR;
						m_zero     = n_iter_zero * MR;
					}

					dcomplex* restrict a_ir = a_ic_use + n_iter_zero * ps_a_use;
					dcomplex* restrict c_ir = c_jr + n_iter_zero * irstep_c;

					/* Ignore the zero region */
					m_off_cblock += m_zero;

					/* Compute the triangular part */
					for( i = m_zero; (i < mc_cur) && ( m_off_cblock < n_off_cblock + nr_cur); i += MR )
					{
						const dim_t mr_cur = (i+MR-1) < mc_cur ? MR : mc_cur - i;

						LOWER_TRIANGLE_OPTIMIZATION_DCOMPLEX()
						{
							gemmsup_ker
							(
							conja,
							conjb,
							mr_cur,
							nr_cur,
							kc_cur,
							alpha_cast,
							a_ir, rs_a_use, cs_a_use,
							b_jr,     rs_b_use, cs_b_use,
							zero,
							ct,     rs_ct,     cs_ct,
							&aux,
							cntx 
							);
							if( col_pref )
							{
								PASTEMAC(z,update_upper_triang)( n_off_cblock, m_off_cblock,
								nr_cur, mr_cur,
								ct, cs_ct, rs_ct,
								beta_use,
								c_ir, cs_c, rs_c );
							}
							else
							{
								PASTEMAC(z,update_lower_triang)( m_off_cblock, n_off_cblock,
								mr_cur, nr_cur,
								ct, rs_ct, cs_ct,
								beta_use,
								c_ir, rs_c, cs_c );
							}
						}

						a_ir += ps_a_use;
						c_ir += irstep_c;
						m_off_cblock += mr_cur;
					}

					/* Invoke the gemmsup millikernel for remaining rectangular part. */
					gemmsup_ker
					(
					  conja,
					  conjb,
					  (i > mc_cur)? 0: mc_cur - i,
					  nr_cur,
					  kc_cur,
					  alpha_cast,
					  a_ir, rs_a_use, cs_a_use,
					  b_jr,     rs_b_use, cs_b_use,
					  beta_use,
					  c_ir,     rs_c,     cs_c,
					  &aux,
					  cntx 
					);

				}
			}

			/* NOTE: This barrier is only needed if we are packing B (since
			   that matrix is packed within the pc loop of this variant). */
			if ( packb ) bli_thread_barrier( thread_pb );
		}
	}

	/* Release any memory that was acquired for packing matrices A and B. */
	PASTEMAC(z,packm_sup_finalize_mem_a)
	(
	  packa,
	  rntm,
	  &mem_a,
	  thread_pa 
	);
	PASTEMAC(z,packm_sup_finalize_mem_b)
	(
	  packb,
	  rntm,
	  &mem_b,
	  thread_pb 
	);

/*
PASTEMAC(z,fprintm)( stdout, "gemmsup_ref_var2: b1", kc_cur, nr_cur, b_jr, rs_b, cs_b, "%4.1f", "" );
PASTEMAC(z,fprintm)( stdout, "gemmsup_ref_var2: a1", mr_cur, kc_cur, a_ir, rs_a, cs_a, "%4.1f", "" );
PASTEMAC(z,fprintm)( stdout, "gemmsup_ref_var2: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%4.1f", "" );
*/
}

void bli_zgemmtsup_u_ref_var2m
     (
       bool             packa,
       bool             packb,
       conj_t           conja,
       conj_t           conjb,
       dim_t            m,
       dim_t            n,
       dim_t            k,
       void*   restrict alpha,
       void*   restrict a, inc_t rs_a, inc_t cs_a,
       void*   restrict b, inc_t rs_b, inc_t cs_b,
       void*   restrict beta,
       void*   restrict c, inc_t rs_c, inc_t cs_c,
       stor3_t          stor_id,
       cntx_t* restrict cntx,
       rntm_t* restrict rntm,
       thrinfo_t* restrict thread 
     )
{
	const num_t dt = PASTEMAC(z,type);

	dcomplex* restrict zero = PASTEMAC(z,0);

	/* If m or n is zero, return immediately. */
	if ( bli_zero_dim2( m, n ) ) return;

	/* If k < 1 or alpha is zero, scale by beta and return. */
	if ( k < 1 || PASTEMAC(z,eq0)( *(( dcomplex* )alpha) ) )
	{
		if ( bli_thread_am_ochief( thread ) )
		{
			PASTEMAC(z,scalm)
			(
			  BLIS_NO_CONJUGATE,
			  0,
			  BLIS_NONUNIT_DIAG,
			  BLIS_DENSE,
			  m, n,
			  beta,
			  c, rs_c, cs_c
			);
		}
		return;
	}

	/* Query the context for various blocksizes. */
	dim_t NR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NR, cntx );
	dim_t MR  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MR, cntx );
	dim_t NC  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_NC, cntx );
	dim_t MC  = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_MC, cntx );
	dim_t KC0 = bli_cntx_get_l3_sup_tri_blksz_def_dt( dt, BLIS_KC, cntx );

	/* Query the maximum blocksize for NR, which implies a maximum blocksize
	   extension for the final iteration. */
	dim_t NRM = bli_cntx_get_l3_sup_tri_blksz_max_dt( dt, BLIS_NR, cntx );

	/* Query the context for the sup microkernel address and cast it to its
	   function pointer type. */
	PASTECH(z,gemmsup_ker_ft)
               gemmsup_ker = bli_cntx_get_l3_sup_tri_ker_dt( dt, stor_id, cntx );

	if( ( 0 == NR ) || ( 0 == MR ) || ( 0 == NC ) || ( 0 == MC ) || ( 0 == KC0 ) )
	{
		NR = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx );
		MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx );
		NC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx );
		MC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx );
		KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx );
		NRM = bli_cntx_get_l3_sup_blksz_max_dt( dt, BLIS_NR, cntx );
		gemmsup_ker = bli_cntx_get_l3_sup_ker_dt( dt, stor_id, cntx );
	}
	const dim_t NRE = NRM - NR;

	dim_t KC;
	if      ( packa && packb )
	{
		KC = KC0;
	}
	else if ( packb )
	{
		if      ( stor_id == BLIS_RRR ||
				  stor_id == BLIS_CCC    ) KC = KC0;
		else if ( stor_id == BLIS_RRC ||
				  stor_id == BLIS_CRC    ) KC = KC0;
		else if ( stor_id == BLIS_RCR ||
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4;
		else                               KC = KC0;
	}
	else if ( packa )
	{
		if      ( stor_id == BLIS_RRR ||
				  stor_id == BLIS_CCC    ) KC = (( KC0 / 2 ) / 2 ) * 2;
		else if ( stor_id == BLIS_RRC ||
				  stor_id == BLIS_CRC    ) KC = KC0;
		else if ( stor_id == BLIS_RCR ||
		          stor_id == BLIS_CCR    ) KC = (( KC0 / 4 ) / 4 ) * 4;
		else                               KC = KC0;
	}
	else /* if ( !packa && !packb ) */
	{
		if      ( stor_id == BLIS_RRR ||
				  stor_id == BLIS_CCC    ) KC = KC0;
		else if ( stor_id == BLIS_RRC ||
				  stor_id == BLIS_CRC    ) KC = KC0;
		else if ( stor_id == BLIS_RCR )
		{
		     if      ( m <=  4*MR ) KC = KC0;
		     else if ( m <= 36*MR ) KC = KC0 / 2;
		     else if ( m <= 56*MR ) KC = (( KC0 / 3 ) / 4 ) * 4;
		     else                   KC = KC0 / 4;
		}
		else if ( m <=   MR && n <=   NR ) KC = KC0;
		else if ( m <= 2*MR && n <= 2*NR ) KC = KC0 / 2;
		else if ( m <= 3*MR && n <= 3*NR ) KC = (( KC0 / 3 ) / 4 ) * 4;
		else if ( m <= 4*MR && n <= 4*NR ) KC = KC0 / 4;
		else                               KC = (( KC0 / 5 ) / 4 ) * 4;
	}

	/* Compute partitioning step values for each matrix of each loop. */
	const inc_t jcstep_c = cs_c;
	const inc_t jcstep_b = cs_b;

	const inc_t pcstep_a = cs_a;
	const inc_t pcstep_b = rs_b;

	const inc_t icstep_c = rs_c;
	const inc_t icstep_a = rs_a;

	const inc_t jrstep_c = cs_c * NR;

	const inc_t irstep_c = rs_c * MR;

	/*
	const inc_t jrstep_b = cs_b * NR;
	( void )jrstep_b;

	const inc_t irstep_c = rs_c * MR;
	const inc_t irstep_a = rs_a * MR;
	*/

	dcomplex ct[ BLIS_STACK_BUF_MAX_SIZE / sizeof( dcomplex ) ] __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE)));

	/* Storage scheme of ct should be same as that of C.
	   Since update routines only support row-major order,
	   col_pref flag is used to induce transpose to matrices before
	   passing to update routine whenever C is col-stored */
	const bool col_pref = (rs_c == 1) ? 1 : 0;

	const inc_t rs_ct = ( col_pref ? 1 : NR );
	const inc_t cs_ct = ( col_pref ? MR : 1 );

	dcomplex* restrict a_00       = a;
	dcomplex* restrict b_00       = b;
	dcomplex* restrict c_00       = c;
	dcomplex* restrict alpha_cast = alpha;
	dcomplex* restrict beta_cast  = beta;

	/* Make local copies of beta and one scalars to prevent any unnecessary
	   sharing of caze lines between the cores' cazes. */
	dcomplex           beta_local = *beta_cast;
	dcomplex           one_local  = *PASTEMAC(z,1);

	auxinfo_t       aux;

	/* Parse and interpret the contents of the rntm_t object to properly
	   set the ways of parallelism for each loop. */
	/*bli_rntm_set_ways_from_rntm_sup( m, n, k, rntm );*/

	/* Initialize a mem_t entry for A and B. Strictly speaking, this is only
	   needed for the matrix we will be packing (if any), but we do it
	   unconditionally to be safe. An alternative way of initializing the
	   mem_t entries is:

	     bli_mem_clear( &mem_a );
	     bli_mem_clear( &mem_b );
	*/
	mem_t mem_a = BLIS_MEM_INITIALIZER;
	mem_t mem_b = BLIS_MEM_INITIALIZER;

	/* Define an array of bszid_t ids, which will act as our substitute for
	   the cntl_t tree. */
	/*                           5thloop  4thloop         packb  3rdloop         packa  2ndloop  1stloop  ukrloop */
	bszid_t bszids_nopack[6] = { BLIS_NC, BLIS_KC,               BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t bszids_packa [7] = { BLIS_NC, BLIS_KC,               BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t bszids_packb [7] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t bszids_packab[8] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR };
	bszid_t* restrict bszids;

	/* Set the bszids pointer to the correct bszids array above based on whiz
	   matrices (if any) are being packed. */
	if ( packa ) { if ( packb ) bszids = bszids_packab;
	               else         bszids = bszids_packa; }
	else         { if ( packb ) bszids = bszids_packb;
	               else         bszids = bszids_nopack; }

	/* Determine whether we are using more than one thread. */
	const bool is_mt = bli_rntm_calc_num_threads( rntm );

	thrinfo_t* restrict thread_jc = NULL;
	thrinfo_t* restrict thread_pc = NULL;
	thrinfo_t* restrict thread_pb = NULL;
	thrinfo_t* restrict thread_ic = NULL;
	thrinfo_t* restrict thread_pa = NULL;
	thrinfo_t* restrict thread_jr = NULL;

	/* Grow the thrinfo_t tree. */
	bszid_t*   restrict bszids_jc = bszids;
	                    thread_jc = thread;
	bli_thrinfo_sup_grow( rntm, bszids_jc, thread_jc );

	/* Compute the JC loop thread range for the current thread. */
	dim_t jc_start, jc_end;
	bli_thread_range_weighted_sub( thread_jc, 0, BLIS_UPPER, m, n, NR, FALSE, &jc_start, &jc_end );
	const dim_t n_local = jc_end - jc_start;

	dim_t m_off = 0;
	dim_t n_off = 0;
	doff_t diagoffc;
	dim_t m_off_cblock, n_off_cblock;
	dim_t jp, j;

	/* Compute number of primary and leftover components of the JC loop. */
	/*const dim_t jc_iter = ( n_local + NC - 1 ) / NC;*/
	const dim_t jc_left =   n_local % NC;

	/* Loop over the n dimension (NC rows/columns at a time). */
	/*for ( dim_t jj = 0; jj < jc_iter; jj += 1 )*/
	for ( dim_t jj = jc_start; jj < jc_end; jj += NC )
	{
		/* Calculate the thread's current JC block dimension. */
		const dim_t nc_cur = ( NC <= jc_end - jj ? NC : jc_left );

		dcomplex* restrict b_jc = b_00 + jj * jcstep_b;
		dcomplex* restrict c_jc = c_00 + jj * jcstep_c;

		/* Grow the thrinfo_t tree. */
		bszid_t*   restrict bszids_pc = &bszids_jc[1];
		                    thread_pc = bli_thrinfo_sub_node( thread_jc );
		bli_thrinfo_sup_grow( rntm, bszids_pc, thread_pc );

		/* Compute the PC loop thread range for the current thread. */
		const dim_t pc_start = 0, pc_end = k;
		const dim_t k_local = k;

		/* Compute number of primary and leftover components of the PC loop. */
		/*const dim_t pc_iter = ( k_local + KC - 1 ) / KC;*/
		const dim_t pc_left =   k_local % KC;

		/* Loop over the k dimension (KC rows/columns at a time). */
		/*for ( dim_t pp = 0; pp < pc_iter; pp += 1 )*/
		for ( dim_t pp = pc_start; pp < pc_end; pp += KC )
		{
			/* Calculate the thread's current PC block dimension. */
			const dim_t kc_cur = ( KC <= pc_end - pp ? KC : pc_left );

			dcomplex* restrict a_pc = a_00 + pp * pcstep_a;
			dcomplex* restrict b_pc = b_jc + pp * pcstep_b;

			/* Only apply beta to the first iteration of the pc loop. */
			dcomplex* restrict beta_use = ( pp == 0 ? &beta_local : &one_local );

			m_off = 0;
			n_off = jj;
			diagoffc = m_off - n_off;

			dcomplex* b_use;
			inc_t  rs_b_use, cs_b_use, ps_b_use;

			/* Set the bszid_t array and thrinfo_t pointer based on whether
			   we will be packing B. If we won't be packing B, we alias to
			   the _pc variables so that code further down can unconditionally
			   reference the _pb variables. Note that *if* we will be packing
			   B, the thrinfo_t node will have already been created by a
			   previous call to bli_thrinfo_grow(), since bszid values of
			   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
			   bszid that is a normal bszid_t value). */
			bszid_t*   restrict bszids_pb;
			if ( packb ) { bszids_pb = &bszids_pc[1];
			               thread_pb = bli_thrinfo_sub_node( thread_pc ); }
			else         { bszids_pb = &bszids_pc[0];
			               thread_pb = thread_pc; }

			/* Determine the packing buffer and related parameters for matrix
			   B. (If B will not be packed, then a_use will be set to point to
			   b and the _b_use strides will be set accordingly.) Then call
			   the packm sup variant chooser, which will call the appropriate
			   implementation based on the schema deduced from the stor_id. */
			PASTEMAC(z,packm_sup_b)
			(
			  packb,
			  BLIS_BUFFER_FOR_B_PANEL, /* This algorithm packs matrix B to */
			  stor_id,                 /* a "panel of B."                  */
			  BLIS_NO_TRANSPOSE,
			  KC,     NC,       /* This "panel of B" is (at most) KC x NC. */
			  kc_cur, nc_cur, NR,
			  &one_local,
			  b_pc,   rs_b,      cs_b,
			  &b_use, &rs_b_use, &cs_b_use,
			                     &ps_b_use,
			  cntx,
			  rntm,
			  &mem_b,
			  thread_pb 
			);

			/* Alias a_use so that it's clear this is our current block of
			   matrix B. */
			dcomplex* restrict b_pc_use = b_use;

			/* We don't need to embed the panel stride of B within the auxinfo_t
			   object because this variant iterates through B in the jr loop,
			   whiz occurs here, within the macrokernel, not within the
			   millikernel. */
			/*bli_auxinfo_set_ps_b( ps_b_use, &aux );*/

			/* Grow the thrinfo_t tree. */
			bszid_t*   restrict bszids_ic = &bszids_pb[1];
			                    thread_ic = bli_thrinfo_sub_node( thread_pb );
			bli_thrinfo_sup_grow( rntm, bszids_ic, thread_ic );

			/* Compute the IC loop thread range for the current thread. */
			dim_t ic_start, ic_end;
			bli_thread_range_weighted_sub( thread_ic, -diagoffc, BLIS_LOWER, nc_cur, m, MR, FALSE, &ic_start, &ic_end );
			const dim_t m_local = ic_end - ic_start;

			/* Compute number of primary and leftover components of the IC loop. */
			/*const dim_t ic_iter = ( m_local + MC - 1 ) / MC;*/
			const dim_t ic_left =   m_local % MC;

			/* Loop over the m dimension (MC rows at a time). */
			/*for ( dim_t ii = 0; ii < ic_iter; ii += 1 )*/
			for ( dim_t ii = ic_start; ii < ic_end; ii += MC )
			{
				/* Calculate the thread's current IC block dimension. */
				dim_t mc_cur = ( MC <= ic_end - ii ? MC : ic_left );

				dim_t nc_pruned = nc_cur;

				m_off = ii;
				n_off = jj;

				if(bli_gemmt_is_strictly_below_diag(m_off, n_off, mc_cur, nc_cur)) continue;

				dcomplex* restrict a_ic = a_pc + ii * icstep_a;
				dcomplex* restrict c_ic = c_jc + ii * icstep_c;

				doff_t diagoffc = m_off - n_off;

				dcomplex* restrict b_pc_pruned = b_pc_use;

				if(diagoffc > 0 )
				{
					jp = diagoffc / NR;
					j = jp * NR;
					nc_pruned = nc_cur - j;
					n_off += j;
					diagoffc = diagoffc % NR;
					c_ic = c_ic + ( j ) * cs_c;
					b_pc_pruned = b_pc_use + ( jp ) * ps_b_use;
				}

				if( ( ( -diagoffc ) + nc_pruned ) < mc_cur )
				{
					mc_cur = -diagoffc + nc_pruned;
				}

				dcomplex* a_use;
				inc_t  rs_a_use, cs_a_use, ps_a_use;

				/* Set the bszid_t array and thrinfo_t pointer based on whether
				   we will be packing B. If we won't be packing A, we alias to
				   the _ic variables so that code further down can unconditionally
				   reference the _pa variables. Note that *if* we will be packing
				   A, the thrinfo_t node will have already been created by a
				   previous call to bli_thrinfo_grow(), since bszid values of
				   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
				   bszid that is a normal bszid_t value). */
				bszid_t*   restrict bszids_pa;
				if ( packa ) { bszids_pa = &bszids_ic[1];
							   thread_pa = bli_thrinfo_sub_node( thread_ic ); }
				else         { bszids_pa = &bszids_ic[0];
							   thread_pa = thread_ic; }

				/* Determine the packing buffer and related parameters for matrix
				   A. (If A will not be packed, then a_use will be set to point to
				   a and the _a_use strides will be set accordingly.) Then call
				   the packm sup variant chooser, which will call the appropriate
				   implementation based on the schema deduced from the stor_id. */
				PASTEMAC(z,packm_sup_a)
				(
				  packa,
				  BLIS_BUFFER_FOR_A_BLOCK, /* This algorithm packs matrix A to */
				  stor_id,                 /* a "block of A."                  */
				  BLIS_NO_TRANSPOSE,
				  MC,     KC,       /* This "block of A" is (at most) MC x KC. */
				  mc_cur, kc_cur, MR,
				  &one_local,
				  a_ic,   rs_a,      cs_a,
				  &a_use, &rs_a_use, &cs_a_use,
				                     &ps_a_use,
				  cntx,
				  rntm,
				  &mem_a,
				  thread_pa 
				);

				/* Alias a_use so that it's clear this is our current block of
				   matrix A. */
				dcomplex* restrict a_ic_use = a_use;

				/* Embed the panel stride of A within the auxinfo_t object. The
				   millikernel will query and use this to iterate through
				   micropanels of A (if needed). */
				bli_auxinfo_set_ps_a( ps_a_use, &aux );

				/* Grow the thrinfo_t tree. */
				bszid_t*   restrict bszids_jr = &bszids_pa[1];
				                    thread_jr = bli_thrinfo_sub_node( thread_pa );
				bli_thrinfo_sup_grow( rntm, bszids_jr, thread_jr );

				/* Compute number of primary and leftover components of the JR loop. */
				dim_t jr_iter = ( nc_pruned + NR - 1 ) / NR;
				dim_t jr_left =   nc_pruned % NR;

				/* Compute the JR loop thread range for the current thread. */
				dim_t jr_start, jr_end;
				bli_thread_range_sub( thread_jr, jr_iter, 1, FALSE, &jr_start, &jr_end );

				/* An optimization: allow the last jr iteration to contain up to NRE
				   columns of C and B. (If NRE > NR, the mkernel has agreed to handle
				   these cases.) Note that this prevents us from declaring jr_iter and
				   jr_left as const. NOTE: We forgo this optimization when packing B
				   since packing an extended edge case is not yet supported. */
				if ( !packb && !is_mt )
				if ( NRE != 0 && 1 < jr_iter && jr_left != 0 && jr_left <= NRE )
				{
					jr_iter--; jr_left += NR;
				}

				/* Loop over the n dimension (NR columns at a time). */
				/*for ( dim_t j = 0; j < jr_iter; j += 1 )*/
				for ( dim_t j = jr_start; j < jr_end; j += 1 )
				{
					const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? NR : jr_left );

					/*
					dcomplex* restrict b_jr = b_pc_use + j * jrstep_b;
					*/
					dcomplex* restrict b_jr = b_pc_pruned + j * ps_b_use;
					dcomplex* restrict c_jr = c_ic     + j * jrstep_c;
					dim_t m_rect = 0;
				        dim_t n_iter_rect = 0;

					m_off_cblock = m_off;
					n_off_cblock = n_off + j * NR;

					if(bli_gemmt_is_strictly_above_diag(m_off_cblock, n_off_cblock, mc_cur, nr_cur))
					{
						m_rect = mc_cur;
					}
					else
					{
						/* calculate the number of rows in rectangular region of the block */
						n_iter_rect = n_off_cblock < m_off_cblock ? 0: (n_off_cblock - m_off_cblock) / MR;
						m_rect = n_iter_rect * MR;
					}

					/* Compute the rectangular part */
					gemmsup_ker
					(
					  conja,
					  conjb,
					  m_rect,
					  nr_cur,
					  kc_cur,
					  alpha_cast,
					  a_ic_use, rs_a_use, cs_a_use,
					  b_jr,     rs_b_use, cs_b_use,
					  beta_use,
					  c_jr,     rs_c,     cs_c,
					  &aux,
					  cntx 
					);

					m_off_cblock = m_off + m_rect;

					dcomplex* restrict a_ir = a_ic_use + n_iter_rect * ps_a_use;
					dcomplex* restrict c_ir = c_jr + n_iter_rect * irstep_c;

					/* compute the remaining triangular part */
					for( dim_t i = m_rect;( i < mc_cur) && (m_off_cblock < n_off_cblock + nr_cur); i += MR )
					{
						const dim_t mr_cur = (i+MR-1) < mc_cur ? MR : mc_cur - i;
						UPPER_TRIANGLE_OPTIMIZATION_DCOMPLEX()
						{
							gemmsup_ker
							(
							conja,
							conjb,
							mr_cur,
							nr_cur,
							kc_cur,
							alpha_cast,
							a_ir, rs_a_use, cs_a_use,
							b_jr,     rs_b_use, cs_b_use,
							zero,
							ct,     rs_ct,     cs_ct, 
							&aux,
							cntx 
							);
	
							if( col_pref )
							{
								PASTEMAC(z,update_lower_triang)( n_off_cblock, m_off_cblock, 
									nr_cur, mr_cur,
									ct, cs_ct, rs_ct,
									beta_use,
									c_ir, cs_c, rs_c );
							}
							else
							{
								PASTEMAC(z,update_upper_triang)( m_off_cblock, n_off_cblock, 
									mr_cur, nr_cur,
									ct, rs_ct, cs_ct,
									beta_use,
									c_ir, rs_c, cs_c );
							}
						}

						a_ir += ps_a_use;
						c_ir += irstep_c;
						m_off_cblock += mr_cur;

					}
				}
			}

			/* NOTE: This barrier is only needed if we are packing B (since
			   that matrix is packed within the pc loop of this variant). */
			if ( packb ) bli_thread_barrier( thread_pb );
		}
	}

	/* Release any memory that was acquired for packing matrices A and B. */
	PASTEMAC(z,packm_sup_finalize_mem_a)
	(
	  packa,
	  rntm,
	  &mem_a,
	  thread_pa 
	);
	PASTEMAC(z,packm_sup_finalize_mem_b)
	(
	  packb,
	  rntm,
	  &mem_b,
	  thread_pb 
	);

/*
PASTEMAC(z,fprintm)( stdout, "gemmsup_ref_var2: b1", kc_cur, nr_cur, b_jr, rs_b, cs_b, "%4.1f", "" );
PASTEMAC(z,fprintm)( stdout, "gemmsup_ref_var2: a1", mr_cur, kc_cur, a_ir, rs_a, cs_a, "%4.1f", "" );
PASTEMAC(z,fprintm)( stdout, "gemmsup_ref_var2: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%4.1f", "" );
*/
}
