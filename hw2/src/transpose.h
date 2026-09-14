#ifndef TRANSPOSE_H
#define TRANSPOSE_H

/*======================================================================
 *  Rectangular real transpose, column-major (first index unit-stride):
 *
 *      A is m x n :  A(i,j) = A[i + j*m]
 *      B is n x m :  B(j,i) = B[j + i*n]
 *      B(j,i) = A(i,j)
 *
 *  In the notation of the Poisson driver this maps
 *  A(1:nx,1:ny) -> B(1:ny,1:nx).
 *
 *  One of the two loops is necessarily non-unit-stride.  The blocked
 *  version keeps a TS x TS tile resident so that both streams are
 *  read and written in cache-line-sized runs.
 *====================================================================*/

/* Tile size in doubles.  8 doubles = 64 bytes = exactly one cache line
   on x86, which measures fastest here across every shape tested; larger
   tiles start evicting one stream before the other is done with it.
   Retune for your machine -- it is the only knob. */
#define TRANSPOSE_TS 8

void transpose_naive  (int m, int n, const double * restrict A,
                                     double * restrict B);
void transpose_blocked(int m, int n, const double * restrict A,
                                     double * restrict B);

/* The one the solver calls. */
void transpose_real   (int m, int n, const double * restrict A,
                                     double * restrict B);
#endif
