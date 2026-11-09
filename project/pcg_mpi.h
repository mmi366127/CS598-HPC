#ifndef PCG_MPI_H
#define PCG_MPI_H

/* Distributed PCG on the same problem as pcg.h (types pcg_prob, pc_t, pcg_info).
 *
 * The n^3 cells are cut into a px x py x pz process grid of blocks (sizes differ by
 * at most one cell when n is not divisible).  Rank = (cx*py + cy)*pz + cz.
 * Local vectors are stored compactly, index (i*nl[1] + j)*nl[2] + k; only the search
 * direction p carries a one-cell ghost layer, filled by a 6-face halo exchange.
 * At physical boundaries the ghost value is g[d] * (adjacent cell), as in pcg.h.
 *
 * Per iteration: 1 halo exchange and 2 allreduces ({p.q} and {r.z, r.r}).
 * Preconditioner PC_SSOR is block-Jacobi SSOR: symmetric Gauss-Seidel inside each
 * rank's block, no coupling across blocks (so iteration counts grow with P). */

#include "pcg.h"

typedef struct {
    pcg_prob P;
    int      dims[3], coord[3];
    int      nl[3], off[3];     /* local extents and global offsets */
    int      lo[3], hi[3];      /* block touches the low / high physical face */
    int      nbr[6];            /* neighbor ranks -x,+x,-y,+y,-z,+z; -1 = physical boundary */
    size_t   N;                 /* local cells */
    double  *sbuf[6], *rbuf[6];
    int      fcount[6];
} pcg_dist;

typedef struct {                /* seconds on this rank, accumulated over one solve */
    double total, halo, apply, vec, prec, reduce;
} pcg_times;

/* Choose a px x py x pz factorization of nranks minimizing the halo surface. */
void pcg_choose_grid(int n, int nranks, int dims[3]);

/* dims[0]*dims[1]*dims[2] must equal num_ranks(). */
void pcg_dist_init(pcg_dist *D, const pcg_prob *P, const int dims[3]);
void pcg_dist_free(pcg_dist *D);

/* f, u: local compact vectors (D->N entries).  u holds the initial guess on entry.
 * Stops when ||r|| <= rtol ||r0|| or after maxit iterations (rtol = 0 forces maxit).
 * hist (optional) receives ||r_k||, k = 0..iters. */
pcg_info pcg_dist_solve(const pcg_dist *D, const double *f, double *u, pc_t pc,
                        double rtol, int maxit, double *hist, pcg_times *tm);

#endif
