#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "poisson_mpi.h"
#include "msg.h"
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

int poisson_plan_mpi_init(poisson_plan_mpi *p,
                           int nx, int ny, double Lx, double Ly,
                           transpose_mpi_strategy strategy)
{
    const int P = num_ranks(), rank = msg_rank();
    int i,j;

    p->nx=nx; p->ny=ny; p->Lx=Lx; p->Ly=Ly;
    p->hx = Lx/(double)(nx+1);
    p->hy = Ly/(double)(ny+1);
    p->strategy = strategy;
    p->lamx=0; p->lamy=0; p->A=0; p->B=0;
    p->verbose = 1;
    memset(&p->t,0,sizeof p->t);

    block_bounds(nx,P,rank,&p->i0x,&p->Mp);
    block_bounds(ny,P,rank,&p->i0y,&p->Qp);

    /* fy contracts along y: local matrix is Mp x ny, batch = Mp. */
    if (block_fst_plan_init(&p->fy,p->Mp,ny)) return 1;
    /* fx contracts along x: local matrix is Qp x nx, batch = Qp. */
    if (block_fst_plan_init(&p->fx,p->Qp,nx)) { block_fst_plan_free(&p->fy); return 1; }

    p->lamx = (double*) malloc((size_t)nx*sizeof(double));
    p->lamy = (double*) malloc((size_t)ny*sizeof(double));
    p->A    = (double*) malloc((size_t)p->Mp*ny*sizeof(double));
    p->B    = (double*) malloc((size_t)p->Qp*nx*sizeof(double));
    if (!p->lamx || !p->lamy || !p->A || !p->B) {
        poisson_plan_mpi_free(p); return 2;
    }

    /* Exact eigenvalues of the 5-point operator (full range -- cheap,
     * O(nx)+O(ny), and both directions need the full range locally at
     * some point: lamx in the divide step, lamy sliced by i0y there). */
    for (i=0; i<nx; ++i) {
        double s = sin(0.5*M_PI*(double)(i+1)/(double)(nx+1));
        p->lamx[i] = 4.0*s*s/(p->hx*p->hx);
    }
    for (j=0; j<ny; ++j) {
        double s = sin(0.5*M_PI*(double)(j+1)/(double)(ny+1));
        p->lamy[j] = 4.0*s*s/(p->hy*p->hy);
    }
    return 0;
}

void poisson_plan_mpi_free(poisson_plan_mpi *p)
{
    block_fst_plan_free(&p->fx);
    block_fst_plan_free(&p->fy);
    free(p->lamx); p->lamx=0;
    free(p->lamy); p->lamy=0;
    free(p->A);    p->A=0;
    free(p->B);    p->B=0;
}

void poisson_solve_mpi(poisson_plan_mpi *p,
                            const double * restrict F, double * restrict U)
{
    const int nx=p->nx, ny=p->ny, Mp=p->Mp, Qp=p->Qp;
    double * restrict A = p->A;
    double * restrict B = p->B;
    poisson_mpi_times *t = &p->t;
    transpose_mpi_prof tp;
    double ta,tb;
    int i,j,k;

    memset(t,0,sizeof *t);
    transpose_mpi_prof_reset();

    ta = msg_wtime();
    for (k=0; k<Mp*ny; ++k) A[k]=F[k];
    t->copy += msg_wtime()-ta;

    const double t0 = msg_wtime();

    /* 1. transform along y: A is Mp x ny, contract over the ny index */
    ta = msg_wtime();
    block_fst_apply(&p->fy,A);
    t->fst_y += msg_wtime()-ta;

    /* 2. distributed transpose: A (Mp x ny, x-block) -> B (Qp x nx, y-block) */
    transpose_mpi(nx,ny,A,B,p->strategy);

    /* 3. transform along x: B is Qp x nx, contract over the nx index */
    ta = msg_wtime();
    block_fst_apply(&p->fx,B);
    t->fst_x += msg_wtime()-ta;

    /* 4. divide by the eigenvalues; B(j,i) = B[j + i*Qp], j local (0..Qp-1)
     *    maps to global y-index i0y+j; i is already global (0..nx-1). */
    ta = msg_wtime();
    for (i=0; i<nx; ++i) {
        const double lx = p->lamx[i];
        double * restrict Bi = &B[(size_t)i*Qp];
        for (j=0; j<Qp; ++j)
            Bi[j] /= (lx + p->lamy[p->i0y + j]);
    }
    t->divide += msg_wtime()-ta;

    /* 5. back along x (S is its own inverse) */
    ta = msg_wtime();
    block_fst_apply(&p->fx,B);
    t->fst_x += msg_wtime()-ta;

    /* 6. distributed transpose back: B (Qp x nx, y-block) -> A (Mp x ny,
     *    x-block).  Args swapped vs. step 2, exactly as the serial code
     *    swaps them in transpose_real(ny,nx,B,A) -- see poisson.c. */
    transpose_mpi(ny,nx,B,A,p->strategy);

    /* 7. back along y */
    ta = msg_wtime();
    block_fst_apply(&p->fy,A);
    t->fst_y += msg_wtime()-ta;

    const double t1 = msg_wtime();

    /* Both transposes are read out of the transpose_mpi counters rather
     * than bracketed here, so the msg/pack/local subdivision lines up
     * exactly with the total. */
    transpose_mpi_prof_get(&tp);
    t->transpose = tp.total;
    t->tr_msg    = tp.msg;
    t->tr_pack   = tp.pack;
    t->tr_local  = tp.local;
    t->total     = t1-t0;

    if (p->verbose && msg_rank()==0) {
        printf("poisson_solve_mpi: Nx %8d  Ny %8d  P %6d  elapsed %12.6f s\n",
               nx+1, ny+1, num_ranks(), t1-t0);
    }

    tb = msg_wtime();
    for (k=0; k<Mp*ny; ++k) U[k]=A[k];
    t->copy += msg_wtime()-tb;
}
