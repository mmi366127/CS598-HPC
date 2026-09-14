/* Naive vs cache-blocked rectangular transpose, A(m x n) -> B(n x m). */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "transpose.h"

static double wall(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec + 1.0e-9*(double)t.tv_nsec;
}

static double bench(void (*f)(int,int,const double*,double*),
                    int m, int n, const double *A, double *B, int nrep)
{
    int k; double t0;
    f(m,n,A,B);                      /* warm up */
    t0=wall();
    for (k=0; k<nrep; ++k) f(m,n,A,B);
    return (wall()-t0)/(double)nrep;
}

static int check(int m, int n, const double *A, const double *B)
{
    int i,j;
    for (j=0; j<n; ++j)
        for (i=0; i<m; ++i)
            if (B[j+i*n] != A[i+j*m]) return 1;
    return 0;
}

int main(int argc, char **argv)
{
    static const int sizes[] = {31,63,127,255,511,1023,2047};
    const int nsz = (int)(sizeof(sizes)/sizeof(sizes[0]));
    int nrep=50, s, i;

    if (argc>1) nrep=atoi(argv[1]);

    printf("rectangular transpose, tile = %d\n\n",TRANSPOSE_TS);
    printf("     m      n      MB   naive (s)  blocked (s)  speedup   GB/s naive  GB/s blkd\n");

    for (s=0; s<nsz; ++s) {
        const int m = sizes[s];
        const int n = sizes[(s+2)%nsz];      /* deliberately rectangular */
        const size_t len = (size_t)m*n;
        const double mb  = 2.0*(double)len*sizeof(double)/1.048576e6;
        double *A=(double*)malloc(len*sizeof(double));
        double *B=(double*)malloc(len*sizeof(double));
        double tn,tb;
        if (!A||!B) return 2;

        for (i=0; i<(int)len; ++i) A[i]=(double)i;

        tn = bench(transpose_naive,  m,n,A,B,nrep);
        if (check(m,n,A,B)) { printf("naive FAILED at %d x %d\n",m,n); return 3; }
        tb = bench(transpose_blocked,m,n,A,B,nrep);
        if (check(m,n,A,B)) { printf("blocked FAILED at %d x %d\n",m,n); return 3; }

        printf("%6d %6d %7.2f  %10.3e  %10.3e  %7.2f  %10.2f %10.2f\n",
               m,n,mb,tn,tb,tn/tb,
               1.0e-9*2.0*(double)len*sizeof(double)/tn,
               1.0e-9*2.0*(double)len*sizeof(double)/tb);
        free(A); free(B);
    }
    printf("\nMB counts both the read and the write stream.\n");
    return 0;
}
