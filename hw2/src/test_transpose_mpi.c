/* Correctness (and basic timing) check for the distributed transpose.
 *
 * Global A is nx x ny with A(i,j) = val(i,j) = i + j*VAL_SCALE, unique
 * per (i,j) so any index-mixup shows up as a wrong value, not just a
 * wrong magnitude.  Each rank builds its own A_local slice directly
 * from the formula (no scatter needed) and checks its own B_local
 * slice against the same formula -- so this also exercises
 * block_bounds() itself.
 *
 * msg.h only exposes a sum-reduction (gsum_double), not a max, so
 * "any rank failed" is folded into a sum of 0/1 flags rather than a
 * true max-error reduction. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "msg.h"
#include "transpose_mpi.h"

#define VAL_SCALE 1000000.0
#define TOL 1e-9

static double val(int i, int j) { return (double)i + (double)j*VAL_SCALE; }

static double *build_A_local(int nx, int ny, int P, int rank)
{
    int i0x,Mp,i,j;
    block_bounds(nx,P,rank,&i0x,&Mp);
    double *A = (double*)malloc((size_t)Mp*ny*sizeof(double));
    for (j=0; j<ny; ++j)
        for (i=0; i<Mp; ++i)
            A[i+j*Mp] = val(i0x+i, j);
    return A;
}

static double check_B_local(int nx, int ny, int P, int rank, const double *B)
{
    int i0y,Qp,i,j;
    double emax = 0.0;
    block_bounds(ny,P,rank,&i0y,&Qp);
    for (i=0; i<nx; ++i)
        for (j=0; j<Qp; ++j) {
            const double e = fabs(B[j+i*Qp] - val(i, i0y+j));
            if (e>emax) emax = e;
        }
    return emax;
}

/* Runs the transpose nrep times, returns this rank's worst-case error
 * over all reps and the best (min) wall time. */
static void run_case(int nx, int ny, int P, int rank,
                      transpose_mpi_strategy strat, int nrep,
                      double *emax_out, double *tbest_out)
{
    int i0y,Qp,r;
    block_bounds(ny,P,rank,&i0y,&Qp);

    double *A = build_A_local(nx,ny,P,rank);
    double *B = (double*)malloc((size_t)Qp*nx*sizeof(double));
    double emax = 0.0, tbest = 1e300;

    for (r=0; r<nrep; ++r) {
        msg_barrier();
        const double t0 = msg_wtime();
        transpose_mpi(nx,ny,A,B,strat);
        const double t1 = msg_wtime();
        if (t1-t0 < tbest) tbest = t1-t0;

        const double e = check_B_local(nx,ny,P,rank,B);
        if (e>emax) emax = e;
    }

    free(A); free(B);
    *emax_out = emax;
    *tbest_out = tbest;
}

int main(int argc, char **argv)
{
    int nx=127, ny=127, nrep=5;

    msg_init(&argc,&argv);
    const int P    = num_ranks();
    const int rank = msg_rank();

    if (argc>1) nx = atoi(argv[1]);
    if (argc>2) ny = atoi(argv[2]);
    if (argc>3) nrep = atoi(argv[3]);

    if (rank==0)
        printf("transpose_mpi correctness: nx=%d ny=%d P=%d\n",nx,ny,P);

    static const char *names[2] = {"dealing-simultaneous","dealing-staggered"};
    static const transpose_mpi_strategy strats[2] =
        { TRANSPOSE_MPI_DEALING_SIMULTANEOUS, TRANSPOSE_MPI_DEALING_STAGGERED };

    int s, any_fail = 0;
    for (s=0; s<2; ++s) {
        double emax, tbest;
        run_case(nx,ny,P,rank,strats[s],nrep,&emax,&tbest);

        const int fail_local = (emax > TOL);
        double flag = (double)fail_local, flag_sum;
        gsum_double(&flag,&flag_sum,1);
        const int fail_any = (flag_sum > 0.5);
        if (fail_any) any_fail = 1;

        if (fail_local)
            printf("  [rank %d] %-8s FAIL max|err| = %.3e\n",
                   rank, names[s], emax);

        if (rank==0)
            printf("  %-8s  best time %10.3e s  %s\n",
                   names[s], tbest, fail_any ? "FAIL" : "PASS");
    }

    if (rank==0)
        printf("\nOVERALL: %s\n", any_fail ? "FAIL" : "PASS");

    msg_finalize();
    return any_fail;
}
