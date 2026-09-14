/* Correctness + timing for the m x n batched FST. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "block_fst.h"

static double wall(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec + 1.0e-9*(double)t.tv_nsec;
}

static double maxdiff(int len, const double *a, const double *b)
{
    int i; double e=0.0;
    for (i=0; i<len; ++i) { double d=fabs(a[i]-b[i]); if (d>e) e=d; }
    return e;
}

int main(int argc, char **argv)
{
    int m=8, n=511, nrep=200, i,k;
    double *X,*Y,*Z,*Yref,t0,dt;

    if (argc>1) m    = atoi(argv[1]);
    if (argc>2) n    = atoi(argv[2]);
    if (argc>3) nrep = atoi(argv[3]);

    block_fst_plan plan;
    if (block_fst_plan_init(&plan,m,n)) return 1;

    X    = (double*)malloc((size_t)m*n*sizeof(double));
    Y    = (double*)malloc((size_t)m*n*sizeof(double));
    Z    = (double*)malloc((size_t)m*n*sizeof(double));
    Yref = (double*)malloc((size_t)m*n*sizeof(double));
    if (!X||!Y||!Z||!Yref) return 2;

    srand(12345);
    for (i=0; i<m*n; ++i) X[i] = 2.0*((double)rand()/(double)RAND_MAX) - 1.0;

    for (i=0; i<m*n; ++i) Y[i]=X[i];
    block_fst_apply(&plan,Y);

    printf("m n nrep                       = %d %d %d\n",m,n,nrep);

    if (n<=1024) {
        block_fst_direct_ortho(m,n,X,Yref);
        printf("max error vs direct FST        = %.6e\n",maxdiff(m*n,Y,Yref));
    }

    /* S*S = I */
    for (i=0; i<m*n; ++i) Z[i]=Y[i];
    block_fst_apply(&plan,Z);
    printf("max error S(SX)-X              = %.6e\n",maxdiff(m*n,Z,X));

    /* matmul reference agrees with the fast path */
    for (i=0; i<m*n; ++i) Z[i]=X[i];
    block_fst_apply_matmul(&plan,Z);
    printf("max error fast vs matmul       = %.6e\n",maxdiff(m*n,Z,Y));

    if (nrep>0) {
        for (i=0; i<m*n; ++i) Y[i]=X[i];
        block_fst_apply(&plan,Y);                 /* warm up */
        t0=wall();
        for (k=0; k<nrep; ++k) block_fst_apply(&plan,Y);
        dt=(wall()-t0)/(double)nrep;
        printf("avg time block_fst_apply       = %.6e sec\n",dt);
        printf("batched FST/s                  = %.6e\n",1.0/dt);
        printf("individual FST/s equiv         = %.6e\n",(double)m/dt);
    }

    block_fst_plan_free(&plan);
    free(X); free(Y); free(Z); free(Yref);
    return 0;
}
