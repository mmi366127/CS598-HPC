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
 * The third strategy is structurally different:
 *
 *   CRYSTAL_ROUTER -- lec06, Fox et al. '88.  ceil(log2(P)) rounds of
 *     recursive bisection: each round every rank does ONE exchange
 *     across the current cut, handing its partner everything bound
 *     for the far side, including blocks it is merely relaying for
 *     other ranks.  Messages per rank fall from P-1 to ~log2(P) at
 *     the cost of forwarding a block up to log2(P) times, so it wins
 *     when the transpose is latency-bound (large P, small blocks) and
 *     loses when it is bandwidth-bound.  See transpose_mpi.c for the
 *     packet format and the non-power-of-two handling.
 */
typedef enum {
    TRANSPOSE_MPI_DEALING_SIMULTANEOUS = 0,
    TRANSPOSE_MPI_DEALING_STAGGERED    = 1,
    TRANSPOSE_MPI_CRYSTAL_ROUTER       = 2
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

/*----------------------------------------------------------------------
 *  Profiling counters.  Accumulated over every transpose_mpi() call
 *  since the last reset, so a caller can bracket a whole solve and
 *  still see where the transpose time went.  These are per-rank wall
 *  times -- reduce across ranks in the caller if a global figure is
 *  wanted.  `total` is the full call; the other three sum to it up to
 *  timer overhead:
 *
 *    msg   -- posting isend/irecv and blocking in msgwait
 *    pack  -- packing/splitting relay buffers (crystal router only;
 *             the dealing variants send straight out of A_local)
 *    local -- transpose_real() on the diagonal block and on each
 *             block that has arrived
 *--------------------------------------------------------------------*/
typedef struct {
    double total, msg, pack, local;
} transpose_mpi_prof;

void transpose_mpi_prof_reset(void);
void transpose_mpi_prof_get(transpose_mpi_prof *out);

#endif
