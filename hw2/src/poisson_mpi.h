#ifndef POISSON_MPI_H
#define POISSON_MPI_H
#include "block_fst.h"
#include "transpose_mpi.h"

/*======================================================================
 *  Parallel counterpart of poisson.h's poisson_plan.  Row-decomposes
 *  the nx x ny grid across P MPI ranks and pivots between the y-FST
 *  and x-FST passes with transpose_mpi() instead of the serial
 *  transpose_real().  poisson.c/poisson.h are left untouched as the
 *  serial reference/baseline.
 *
 *  DECOMPOSITION (matches transpose_mpi.h exactly):
 *
 *    before the transpose: rank p owns Mp x-rows, Ip = block_bounds(nx,P,p),
 *                           all ny columns          -- A is Mp x ny
 *    after  the transpose: rank p owns Qp y-rows, Jp = block_bounds(ny,P,p),
 *                           all nx columns           -- B is Qp x nx
 *
 *  F and U passed to poisson_solve_mpi are this rank's LOCAL Mp x ny
 *  slice (x-rows Ip, all of ny) -- the caller is responsible for
 *  building only its own slice (see test_poisson_mpi.c), not the
 *  full nx x ny array.
 *====================================================================*/

typedef struct {
    int    nx, ny;                 /* GLOBAL grid size                  */
    double Lx, Ly, hx, hy;
    int    i0x, Mp;                 /* this rank's x-range, before transpose */
    int    i0y, Qp;                 /* this rank's y-range, after transpose  */
    block_fst_plan fx;              /* m = Qp, n = nx : contracts along x */
    block_fst_plan fy;              /* m = Mp, n = ny : contracts along y */
    double *lamx, *lamy;            /* full-length, nx and ny             */
    double *A, *B;                  /* Mp x ny  and  Qp x nx local scratch */
    transpose_mpi_strategy strategy;
} poisson_plan_mpi;

int  poisson_plan_mpi_init(poisson_plan_mpi *p,
                            int nx, int ny, double Lx, double Ly,
                            transpose_mpi_strategy strategy);
void poisson_plan_mpi_free(poisson_plan_mpi *p);

/* F and U are LOCAL Mp x ny arrays (x-rows Ip, all of ny); U may alias F. */
void poisson_solve_mpi(poisson_plan_mpi *p,
                            const double * restrict F, double * restrict U);
#endif
