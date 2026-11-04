#ifndef PCG_H
#define PCG_H

/* Serial PCG for  -Laplace(u) = f  on [0,L]^3, 7-point finite differences on an
 * n x n x n grid, spacing h.  Index: idx(i,j,k) = (i*n + j)*n + k, k fastest.
 *
 * Boundary treatment is a ghost value next to each face: ghost = g[d] * u_adjacent
 * (+ a constant, which the caller moves to the rhs).  One g per direction d = x,y,z
 * (same on both faces of that direction):
 *    g =  0  homogeneous Dirichlet on a vertex-centered grid (u_{-1} = 0, h = L/(n+1))
 *    g = -1  Dirichlet on a cell-centered grid (face at h/2: ghost = 2 u_b - u),
 *            caller adds 2 u_b / h^2 to the rhs of the cells touching that face
 *    g = +1  zero normal derivative on a cell-centered grid (ghost = u)
 * The diagonal is (6 - sum of g over the faces the cell touches) / h^2. */

typedef struct {
    int    n;
    double h;
    double g[3];
} pcg_prob;

typedef enum { PC_NONE = 0, PC_JACOBI, PC_SSOR } pc_t;

typedef struct {
    int    iters;      /* iterations performed */
    double res0;       /* ||r0||_2 */
    double res;        /* final ||r||_2 */
    int    converged;
} pcg_info;

/* y = A x */
void apply_A(const pcg_prob *P, const double *x, double *y);

/* Solve A u = f. u is the initial guess on entry (use zeros), solution on exit.
 * Stops when ||r|| <= rtol * ||r0|| or after maxit iterations.
 * If hist != NULL it receives ||r_k|| for k = 0..iters (size maxit+1). */
pcg_info pcg_solve(const pcg_prob *P, const double *f, double *u,
                   pc_t pc, double rtol, int maxit, double *hist);

#endif
