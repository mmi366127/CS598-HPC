#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "poisson.h"
#include "transpose.h"
#include "msg.h"
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

int poisson_plan_init(poisson_plan *p,
                           int nx, int ny, double Lx, double Ly)
{
    int i,j;

    p->nx=nx; p->ny=ny; p->Lx=Lx; p->Ly=Ly;
    p->hx = Lx/(double)(nx+1);
    p->hy = Ly/(double)(ny+1);
    p->lamx=0; p->lamy=0; p->A=0; p->B=0;

    /* fy contracts along y: the matrix is nx x ny, batch = nx. */
    if (block_fst_plan_init(&p->fy,nx,ny)) return 1;
    /* fx contracts along x: the matrix is ny x nx, batch = ny. */
    if (block_fst_plan_init(&p->fx,ny,nx)) { block_fst_plan_free(&p->fy); return 1; }

    p->lamx = (double*) malloc((size_t)nx*sizeof(double));
    p->lamy = (double*) malloc((size_t)ny*sizeof(double));
    p->A    = (double*) malloc((size_t)nx*ny*sizeof(double));
    p->B    = (double*) malloc((size_t)nx*ny*sizeof(double));
    if (!p->lamx || !p->lamy || !p->A || !p->B) {
        poisson_plan_free(p); return 2;
    }

    /* Exact eigenvalues of the 5-point operator. */
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

void poisson_plan_free(poisson_plan *p)
{
    block_fst_plan_free(&p->fx);
    block_fst_plan_free(&p->fy);
    free(p->lamx); p->lamx=0;
    free(p->lamy); p->lamy=0;
    free(p->A);    p->A=0;
    free(p->B);    p->B=0;
}

void poisson_solve(poisson_plan *p,
                        const double * restrict F, double * restrict U)
{
    const int nx=p->nx, ny=p->ny;
    double * restrict A = p->A;
    double * restrict B = p->B;
    int i,j,k;

    for (k=0; k<nx*ny; ++k) A[k]=F[k];

    /* ---- timed region: the actual FST solve, steps 1-7 -------------- */
    const double t0 = msg_wtime();

    /* 1. transform along y: A is nx x ny, contract over the ny index */
    block_fst_apply(&p->fy,A);

    /* 2. A(1:nx,1:ny) -> B(1:ny,1:nx) */
    transpose_real(nx,ny,A,B);
    

    /* 3. transform along x: B is ny x nx, contract over the nx index */
    block_fst_apply(&p->fx,B);

    /* 4. divide by the eigenvalues;  B(j,i) = B[j + i*ny] */
    for (i=0; i<nx; ++i) {
        const double lx = p->lamx[i];
        double * restrict Bi = &B[i*ny];
        for (j=0; j<ny; ++j)
            Bi[j] /= (lx + p->lamy[j]);
    }

    /* 5. back along x (S is its own inverse) */
    block_fst_apply(&p->fx,B);

    /* 6. B(1:ny,1:nx) -> A(1:nx,1:ny) */
    transpose_real(ny,nx,B,A);

    /* 7. back along y */
    block_fst_apply(&p->fy,A);

    const double t1 = msg_wtime();
    /* ---- end timed region -------------------------------------------- */

    if (msg_rank()==0) {
        const int    Nx      = nx+1, Ny = ny+1;
        const double elapsed = t1-t0;
        /* 4 batched-FST passes (steps 1,3,5,7), each O(m n log2 n);
         * the crude count below folds all four into one estimate:
         * flops ~ 2*10*nx*ny*(log2(Nx)+log2(Ny)). */
        const double flops  = 2.0*10.0*(double)nx*(double)ny*
                               (log2((double)Nx) + log2((double)Ny));
        const double gflops = (flops/elapsed)/1e9;
        printf("poisson_solve: Nx %8d  Ny %8d  elapsed %12.6f s  "
               "GFLOPS %8.3f\n", Nx, Ny, elapsed, gflops);
    }

    for (k=0; k<nx*ny; ++k) U[k]=A[k];
}

/* F = -Laplacian_h U, zero Dirichlet data outside the index range. */
void poisson_residual_op(const poisson_plan *p,
                              const double * restrict U,
                              double * restrict F)
{
    const int nx=p->nx, ny=p->ny;
    const double cx=1.0/(p->hx*p->hx), cy=1.0/(p->hy*p->hy);
    int i,j;

    for (j=0; j<ny; ++j) {
        for (i=0; i<nx; ++i) {
            const double c  = U[i + j*nx];
            const double w  = (i>0)    ? U[(i-1) + j*nx] : 0.0;
            const double e  = (i<nx-1) ? U[(i+1) + j*nx] : 0.0;
            const double s  = (j>0)    ? U[i + (j-1)*nx] : 0.0;
            const double nn = (j<ny-1) ? U[i + (j+1)*nx] : 0.0;
            F[i + j*nx] = cx*(2.0*c - w - e) + cy*(2.0*c - s - nn);
        }
    }
}
