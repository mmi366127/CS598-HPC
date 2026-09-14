#ifndef BLOCK_FST_H
#define BLOCK_FST_H
#include "block_fft.h"

/*======================================================================
 *  Batched orthonormal DST-I (FST) on an m x n matrix.
 *
 *  STORAGE CONVENTION (used everywhere in this package)
 *  ----------------------------------------------------
 *  All 2-D arrays are column-major with the FIRST index unit-stride,
 *  i.e. Fortran ordering A(i,j):
 *
 *      A(i,j)  ==  A[i + j*m],    i = 0..m-1  (unit stride)
 *                                 j = 0..n-1  (stride m)
 *
 *  block_fst_apply() CONTRACTS over the second ("n") index and
 *  VECTORIZES over the first, unit-stride ("m") index:
 *
 *      A(i,k)  <-  sum_j  S_n(k,j) * A(i,j)
 *
 *  so the inner loop runs over i and is unit-stride in both operands.
 *  The m columns are the batch; n is the transform length.
 *
 *  S_n is the orthonormal DST-I matrix of order n,
 *
 *      S_n(k,j) = sqrt(2/(n+1)) * sin(pi*(j+1)*(k+1)/(n+1)),
 *
 *  which satisfies S^T = S and S*S = I, so the same routine is its own
 *  inverse.  Requires L = 2*(n+1) = 4^k, i.e. n = 31, 127, 511, 2047...
 *====================================================================*/

typedef struct {
    int m;                  /* batch length, unit-stride direction   */
    int n;                  /* transform length, contracted direction*/
    int L;                  /* 2*(n+1), must be a power of four      */
    block_fft_plan fft;
    cpx    *work;           /* L x m complex scratch                 */
    double *S;              /* optional explicit n x n sine matrix   */
    double *tmp;            /* m x n real scratch                    */
} block_fst_plan;

int  block_fst_plan_init(block_fst_plan *p, int m, int n);
void block_fst_plan_free(block_fst_plan *p);

/* In-place FST of the m x n matrix A.  Self-inverse. */
void block_fst_apply(block_fst_plan *p, double * restrict A);

/* Explicit-matrix references, same in-place semantics. */
int  block_fst_build_matrix(block_fst_plan *p);
void block_fst_apply_slow  (block_fst_plan *p, double * restrict A);
void block_fst_apply_matmul(block_fst_plan *p, double * restrict A);

/* O(m n^2) out-of-place reference, straight from the definition. */
void block_fst_direct_ortho(int m, int n,
                            const double * restrict A,
                            double * restrict Y);
#endif
