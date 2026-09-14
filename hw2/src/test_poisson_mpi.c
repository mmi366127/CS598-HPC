/* Parallel counterpart of test_poisson.c's continuous-convergence check.
 *
 * Only that check is ported here: it needs just the analytic formulas
 * rhs_f/exact_u at each rank's own (x,y), no cross-rank data, so it
 * decomposes cleanly.  test_poisson.c's *other* check --
 * poisson_residual_op building F from a random U -- evaluates a 5-point
 * stencil across x-neighbors, which cross rank boundaries under this
 * x-row decomposition; that needs a halo/ghost exchange (a different,
 * nearest-neighbor communication pattern) and is left to the serial
 * test for now. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "poisson_mpi.h"
#include "msg.h"
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static double exact_u(double x, double y, double Lx, double Ly)
{
    return sin(M_PI*x/Lx)*sin(2.0*M_PI*y/Ly);
}

static double rhs_f(double x, double y, double Lx, double Ly)
{
    const double c = M_PI*M_PI/(Lx*Lx) + 4.0*M_PI*M_PI/(Ly*Ly);
    return c*exact_u(x,y,Lx,Ly);
}

int main(int argc, char **argv)
{
    int Nx=128, Ny=512, i,j;
    double Lx=2.0, Ly=1.0;
    poisson_plan_mpi p;
    transpose_mpi_strategy strat = TRANSPOSE_MPI_DEALING_STAGGERED;

    msg_init(&argc,&argv);
    const int rank = msg_rank(), P = num_ranks();

    if (argc>1) Nx = atoi(argv[1]);
    if (argc>2) Ny = atoi(argv[2]);
    if (argc>3) Lx = atof(argv[3]);
    if (argc>4) Ly = atof(argv[4]);
    if (argc>5) strat = atoi(argv[5]) ? TRANSPOSE_MPI_DEALING_STAGGERED
                                       : TRANSPOSE_MPI_DEALING_SIMULTANEOUS;
    const int nx=Nx-1, ny=Ny-1;

    if (poisson_plan_mpi_init(&p,nx,ny,Lx,Ly,strat)) return 1;

    if (rank==0) {
        printf("running on %d rank(s), strategy=%s\n", P,
               strat==TRANSPOSE_MPI_DEALING_STAGGERED ? "dealing-staggered"
                                                        : "dealing-simultaneous");
        printf("domain  [0,%g] x [0,%g]\n",Lx,Ly);
        printf("nx ny   = %d %d      (Nx Ny = %d %d)\n",nx,ny,Nx,Ny);
        printf("hx hy   = %.6e %.6e\n",p.hx,p.hy);
    }

    double *F  =(double*)malloc((size_t)p.Mp*ny*sizeof(double));
    double *U  =(double*)malloc((size_t)p.Mp*ny*sizeof(double));
    double *Uex=(double*)malloc((size_t)p.Mp*ny*sizeof(double));
    if (!F||!U||!Uex) return 2;

    /* LOCAL slice only: rank owns x-rows [i0x, i0x+Mp), all of ny.
     * (This is the parallel analogue of test_poisson.c's loop over the
     * full i=0..nx-1 range -- offset by i0x, bounded by Mp instead.) */
    for (j=0; j<ny; ++j) {
        double y=(double)(j+1)*p.hy;
        for (i=0; i<p.Mp; ++i) {
            double x=(double)(p.i0x+i+1)*p.hx;
            F  [i+j*p.Mp]=rhs_f  (x,y,Lx,Ly);
            Uex[i+j*p.Mp]=exact_u(x,y,Lx,Ly);
        }
    }

    poisson_solve_mpi(&p,F,U);   /* warm-up, same rationale as test_poisson.c */
    poisson_solve_mpi(&p,F,U);

    double emax=0.0;
    for (i=0; i<p.Mp*ny; ++i) {
        const double e = fabs(U[i]-Uex[i]);
        if (e>emax) emax=e;
    }

    /* msg.h has no max-reduce, only gsum_double (sum) -- for this
     * one-off diagnostic, just gather each rank's local max to rank 0
     * by hand instead of adding a collective for a single printf. */
    if (rank==0) {
        double gmax = emax;
        int r;
        for (r=1; r<P; ++r) {
            double v;
            irecv(r,&v,(int)sizeof v,99);
            msgwait();
            if (v>gmax) gmax=v;
        }
        printf("continuous  max|U - u(x,y)| (global) = %.6e  (expect O(h^2))\n",gmax);
    } else {
        isend(0,&emax,(int)sizeof emax,99);
        msgwait();
    }

    poisson_plan_mpi_free(&p);
    free(F); free(U); free(Uex);
    msg_finalize();
    return 0;
}
