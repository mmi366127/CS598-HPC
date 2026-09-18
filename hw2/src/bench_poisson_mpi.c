/* Sweep driver for the distributed Poisson solve.
 *
 * One (Nx, Ny, P, strategy) point per invocation; the sweep over the
 * grid lives in poisson.sbatch.  Emits one machine-readable "BENCH"
 * line so the whole sweep can be tabulated from the job log, plus a
 * human-readable block.
 *
 * Reductions: msg.h exposes only gsum_double, so the max-over-ranks
 * that the timings need is done by hand -- every rank ships its vector
 * to rank 0, which reduces it there (same tactic test_poisson_mpi.c
 * already uses for its one error value, just vectorised).
 *
 * Timings are reported as MAX over ranks, which is the figure that
 * matters: the solve is only finished when the slowest rank is, and a
 * mean would hide exactly the load imbalance the block decomposition
 * introduces when P does not divide nx.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "poisson_mpi.h"
#include "msg.h"
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

#define NT 9   /* timing entries gathered per rank, see pack_times() */

static double exact_u(double x, double y, double Lx, double Ly)
{
    return sin(M_PI*x/Lx)*sin(2.0*M_PI*y/Ly);
}

static double rhs_f(double x, double y, double Lx, double Ly)
{
    const double c = M_PI*M_PI/(Lx*Lx) + 4.0*M_PI*M_PI/(Ly*Ly);
    return c*exact_u(x,y,Lx,Ly);
}

static void pack_times(const poisson_mpi_times *t, double *v)
{
    v[0]=t->total;   v[1]=t->fst_y;   v[2]=t->fst_x;
    v[3]=t->transpose; v[4]=t->tr_msg; v[5]=t->tr_pack;
    v[6]=t->tr_local; v[7]=t->divide; v[8]=t->copy;
}

/* Gather every rank's n-vector onto rank 0 and hand back the one
 * belonging to the CRITICAL-PATH rank (largest entry 0, the total),
 * along with the smallest total seen.
 *
 * Taking a per-component max instead would mix components measured on
 * different ranks, and the parts would not add up to the whole; this
 * way the reported breakdown is one rank's self-consistent accounting
 * of the solve that everyone else was waiting on.  `tmin` is returned
 * so the spread -- i.e. the load imbalance when P does not divide nx
 * -- stays visible next to it.
 *
 * Rank 0 posts every irecv before the single msgwait, so this is one
 * round of messages, not P-1 sequential ones. */
static void reduce_critical(const double *in, double *out, int n, double *tmin)
{
    const int P = num_ranks(), rank = msg_rank();
    int r,imax=0;

    if (rank!=0) {
        isend(0,in,(int)((size_t)n*sizeof(double)),7);
        msgwait();
        return;
    }
    double *all = (double*)malloc((size_t)n*P*sizeof(double));
    memcpy(all,in,(size_t)n*sizeof(double));
    for (r=1; r<P; ++r)
        irecv(r,all+(size_t)r*n,(int)((size_t)n*sizeof(double)),7);
    msgwait();

    *tmin = all[0];
    for (r=1; r<P; ++r) {
        const double v = all[(size_t)r*n];
        if (v > all[(size_t)imax*n]) imax = r;
        if (v < *tmin) *tmin = v;
    }
    memcpy(out, all+(size_t)imax*n, (size_t)n*sizeof(double));
    free(all);
}

/* Max of a single scalar over all ranks, result on rank 0. */
static void reduce_max1(double in, double *out)
{
    const int P = num_ranks(), rank = msg_rank();
    int r;
    if (rank!=0) { isend(0,&in,(int)sizeof in,8); msgwait(); return; }
    double *all = (double*)malloc((size_t)P*sizeof(double));
    all[0]=in;
    for (r=1; r<P; ++r) irecv(r,all+r,(int)sizeof(double),8);
    msgwait();
    *out = all[0];
    for (r=1; r<P; ++r) if (all[r] > *out) *out = all[r];
    free(all);
}

int main(int argc, char **argv)
{
    int Nx=128, Ny=128, nrep=3, i,j,r;
    double Lx=2.0, Ly=1.0;
    poisson_plan_mpi p;
    transpose_mpi_strategy strat = TRANSPOSE_MPI_DEALING_STAGGERED;
    static const char *sname[3] =
        {"dealing-simultaneous","dealing-staggered","crystal-router"};

    msg_init(&argc,&argv);
    const int rank = msg_rank(), P = num_ranks();

    if (argc>1) Nx   = atoi(argv[1]);
    if (argc>2) Ny   = atoi(argv[2]);
    if (argc>3) {
        const int k = atoi(argv[3]);
        strat = (k==2) ? TRANSPOSE_MPI_CRYSTAL_ROUTER
              : (k==1) ? TRANSPOSE_MPI_DEALING_STAGGERED
                       : TRANSPOSE_MPI_DEALING_SIMULTANEOUS;
    }
    if (argc>4) nrep = atoi(argv[4]);
    const int nx=Nx-1, ny=Ny-1;

    /* block_bounds would hand some rank an empty slice, which
     * block_fst_plan_init rejects (it requires m >= 1). */
    if (P > nx || P > ny) {
        if (rank==0)
            printf("BENCH %6d %6d %4d %d  SKIPPED (P > nx: some rank would own no rows)\n",
                   Nx,Ny,P,(int)strat);
        msg_finalize();
        return 0;
    }

    if (poisson_plan_mpi_init(&p,nx,ny,Lx,Ly,strat)) {
        if (rank==0) printf("BENCH %6d %6d %4d %d  PLAN-INIT-FAILED\n",Nx,Ny,P,(int)strat);
        msg_finalize();
        return 1;
    }
    p.verbose = 0;   /* this driver does its own reporting */

    double *F  =(double*)malloc((size_t)p.Mp*ny*sizeof(double));
    double *U  =(double*)malloc((size_t)p.Mp*ny*sizeof(double));
    double *Uex=(double*)malloc((size_t)p.Mp*ny*sizeof(double));
    if (!F||!U||!Uex) { if(rank==0) printf("BENCH %d %d %d %d OOM\n",Nx,Ny,P,(int)strat); return 2; }

    /* This rank's slice only: x-rows [i0x, i0x+Mp), all of ny. */
    for (j=0; j<ny; ++j) {
        const double y=(double)(j+1)*p.hy;
        for (i=0; i<p.Mp; ++i) {
            const double x=(double)(p.i0x+i+1)*p.hx;
            F  [i+j*p.Mp]=rhs_f  (x,y,Lx,Ly);
            Uex[i+j*p.Mp]=exact_u(x,y,Lx,Ly);
        }
    }

    /* Warm-up (first-touch, plan tables, MPI connection setup), then
     * nrep timed reps keeping the breakdown of the FASTEST rep -- the
     * least noisy sample, matching the "best time" convention used by
     * test_transpose_mpi. */
    poisson_solve_mpi(&p,F,U);

    double vbest[NT] = {0}; double best=1e300;
    for (r=0; r<nrep; ++r) {
        double v[NT];
        msg_barrier();
        poisson_solve_mpi(&p,F,U);
        pack_times(&p.t,v);
        /* Rank 0's total decides which rep is "the fastest" so that
         * every rank keeps the breakdown of the SAME rep. */
        double sel = v[0];
        bcast(&sel,(int)sizeof sel,0);
        if (sel < best) { best = sel; memcpy(vbest,v,sizeof v); }
    }

    double tmax[NT] = {0}, tmin = 0.0;
    reduce_critical(vbest,tmax,NT,&tmin);

    double emax=0.0;
    for (i=0; i<p.Mp*ny; ++i) {
        const double e = fabs(U[i]-Uex[i]);
        if (e>emax) emax=e;
    }
    double gerr = 0.0;
    reduce_max1(emax,&gerr);

    if (rank==0) {
        /* Same flop model as the serial poisson_solve() in poisson.c:
         * 4 batched-FST passes, each O(m n log2 n), folded into
         * 2*10*nx*ny*(log2 Nx + log2 Ny) so the GFLOPS numbers here and
         * there are directly comparable. */
        const double flops  = 2.0*10.0*(double)nx*(double)ny*
                               (log2((double)Nx)+log2((double)Ny));
        const double gflops = (flops/tmax[0])/1e9;

        printf("BENCH %6d %6d %4d %d %-20s "
               "total %.6e tmin %.6e fst_y %.6e fst_x %.6e transpose %.6e "
               "tr_msg %.6e tr_pack %.6e tr_local %.6e divide %.6e copy %.6e "
               "err %.6e gflops %.4f\n",
               Nx,Ny,P,(int)strat,sname[(int)strat],
               tmax[0],tmin,tmax[1],tmax[2],tmax[3],
               tmax[4],tmax[5],tmax[6],tmax[7],tmax[8],
               gerr,gflops);
        fflush(stdout);
    }

    poisson_plan_mpi_free(&p);
    free(F); free(U); free(Uex);
    msg_finalize();
    return 0;
}
