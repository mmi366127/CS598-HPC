#ifndef TRANSPOSE_MPI_H
#define TRANSPOSE_MPI_H

/*======================================================================
 *  Distributed rectangular transpose between two row-block
 *  decompositions, built from isend/irecv pairs (see msg.h).
 *
 *  Global A is nx x ny, column-major, x unit-stride: A(i,j) = A[i+j*nx].
 *  Global B is ny x nx, column-major, y unit-stride: B(j,i) = A(i,j).
 *
 *  Rank p owns:
 *    A_local : Mp x ny   (Mp x-rows Ip = block_bounds(nx,P,p), all of ny)
 *    B_local : Qp x nx   (Qp y-rows Jp = block_bounds(ny,P,p), all of nx)
 *
 *  transpose_mpi() takes A_local on every rank and produces B_local on
 *  every rank -- the classic all-to-all pencil transpose used to pivot
 *  between the y-FST pass and the x-FST pass of the Poisson solve.
 *====================================================================*/

/*
 * Both strategies below are the direct point-to-point all-to-all --
 * the "Dealing" algorithm (lec05, p.5-8): every rank exchanges its own
 * distinct P-1 blocks straight with their owners, no intermediary
 * hops.  They differ only in WHEN the P-1 messages hit the network:
 *
 *   DEALING_SIMULTANEOUS -- all P-1 isend/irecv posted at once, one
 *     msgwait.  This is the original version lec05 p.6 flags as
 *     having a hotspot problem (every rank's send loop reaches
 *     destination q=0 on the same iteration).
 *
 *   DEALING_STAGGERED -- P rounds, one partner per round via the
 *     "circle method" partner(p,r) = (r-p) mod P (self-inverse, so
 *     both sides agree on the partner), each round closed out with
 *     its own msgwait before the next begins.  This is the real-time
 *     equivalent of lec05 p.6's fix: since msg.h only exposes
 *     non-blocking isend/irecv plus a single global msgwait, posting
 *     order alone can't stagger traffic -- an actual msgwait per
 *     round is what forces the rounds apart in time.
 *
 * A third, structurally different algorithm -- the Crystal Router
 * (lec06, Fox et al. '88): log2(P) rounds of recursive bisection
 * exchange instead of P direct exchanges -- is not implemented here.
 */
typedef enum {
    TRANSPOSE_MPI_DEALING_SIMULTANEOUS = 0,
    TRANSPOSE_MPI_DEALING_STAGGERED    = 1
} transpose_mpi_strategy;

/* Split n items across P ranks as evenly as possible: rank p gets
 * `count` items starting at `start`.  The first (n % P) ranks get one
 * extra item.  Used for both the initial x-split and the post-
 * transpose y-split. */
void block_bounds(int n, int P, int p, int *start, int *count);

void transpose_mpi(int nx, int ny,
                    const double * restrict A_local,
                    double * restrict B_local,
                    transpose_mpi_strategy strategy);

#endif
