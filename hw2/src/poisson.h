#ifndef POISSON_H
#define POISSON_H
#include "block_fst.h"

/*======================================================================
 *  Dirichlet Poisson solver on the rectangle [0,Lx] x [0,Ly]:
 *
 *      -Laplacian u = f   in the interior,   u = 0 on the boundary,
 *
 *  discretized with the standard 5-point stencil.
 *
 *  GRID.  Nx = nx+1 uniform intervals in x, so
 *
 *      hx = Lx/Nx = Lx/(nx+1),   x_i = i*hx,  i = 1..nx  (interior)
 *      hy = Ly/Ny = Ly/(ny+1),   y_j = j*hy,  j = 1..ny
 *
 *  nx and ny are therefore the numbers of interior unknowns, and the
 *  arrays are nx x ny.  Each must satisfy 2*(n+1) = 4^k, so
 *  nx, ny in {31, 127, 511, 2047, ...} -- they need not be equal.
 *
 *  STORAGE.  Column-major, x unit-stride:  F(i,j) = F[i + j*nx].
 *
 *  METHOD.  Diagonalize in both directions with the orthonormal DST-I:
 *
 *      A  = F                                  (nx x ny)
 *      A <- S_ny applied along y               (contract 2nd index)
 *      B  = A^T                                (ny x nx)
 *      B <- S_nx applied along x               (contract 2nd index)
 *      B(j,i) /= lam_x(i) + lam_y(j)
 *      B <- S_nx along x        (S is its own inverse)
 *      A  = B^T                                (nx x ny)
 *      A <- S_ny along y
 *      U  = A
 *
 *  with the exact eigenvalues of the 5-point operator,
 *
 *      lam_x(i) = (4/hx^2) sin^2( pi*(i+1) / (2*(nx+1)) ).
 *
 *  Because those are the eigenvalues of the DISCRETE operator, the
 *  solver inverts it exactly (to roundoff); the O(h^2) error you see
 *  against a continuous solution is pure truncation error.
 *====================================================================*/

typedef struct {
    int    nx, ny;
    double Lx, Ly, hx, hy;
    block_fst_plan fx;      /* m = ny, n = nx : contracts along x */
    block_fst_plan fy;      /* m = nx, n = ny : contracts along y */
    double *lamx, *lamy;
    double *A, *B;          /* nx x ny and ny x nx scratch */
} poisson_plan;

int  poisson_plan_init(poisson_plan *p,
                            int nx, int ny, double Lx, double Ly);
void poisson_plan_free(poisson_plan *p);

/* U and F are nx x ny; U may alias F. */
void poisson_solve(poisson_plan *p,
                        const double * restrict F, double * restrict U);

/* Apply the 5-point operator -Laplacian_h, homogeneous Dirichlet. */
void poisson_residual_op(const poisson_plan *p,
                              const double * restrict U,
                              double * restrict F);
#endif
