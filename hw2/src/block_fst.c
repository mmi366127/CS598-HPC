#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "block_fst.h"
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static int is_power_four(int L)
{
    int t=1;
    while (t<L) t*=4;
    return t==L;
}

int block_fst_plan_init(block_fst_plan *p, int m, int n)
{
    p->m=m; p->n=n; p->L=2*(n+1); p->work=0; p->S=0; p->tmp=0;
    if (m<1 || n<1) {
        fprintf(stderr,"block_fst: require m,n >= 1\n");
        return 1;
    }
    if (!is_power_four(p->L)) {
        fprintf(stderr,"block_fst: require L=2*(n+1)=4^k; try n=31,127,511,2047,...\n");
        return 1;
    }
    p->work = (cpx*)    malloc((size_t)p->L*m*sizeof(cpx));
    p->tmp  = (double*) malloc((size_t)m*n*sizeof(double));
    if (!p->work || !p->tmp) { block_fst_plan_free(p); return 2; }
    if (block_fft_plan_init(&p->fft,p->L,m,4)) {
        block_fst_plan_free(p); return 3;
    }
    return 0;
}

void block_fst_plan_free(block_fst_plan *p)
{
    block_fft_plan_free(&p->fft);
    free(p->work); p->work=0;
    free(p->S);    p->S=0;
    free(p->tmp);  p->tmp=0;
}

/*----------------------------------------------------------------------
 *  Fast path: odd extension of length L=2(n+1) plus one batched radix-4
 *  FFT.  The extension is built in the FFT's own (L x m) layout, which
 *  is the same column-major convention as A.
 *--------------------------------------------------------------------*/
void block_fst_apply(block_fst_plan *p, double * restrict A)
{
    const int m=p->m, n=p->n, N=n+1, L=p->L;
    int i,j,k;
    const double scale = -0.5*sqrt(2.0/(double)N);

    for (i=0; i<L*m; ++i) { p->work[i].re=0.0; p->work[i].im=0.0; }

    /* odd extension:  w(j) = A(:,j-1),  w(N+j) = -A(:,n-j)  */
    for (j=1; j<=n; ++j) {
        cpx * restrict wj  = &p->work[j*m];
        cpx * restrict wnj = &p->work[(N+j)*m];
        const double * restrict aj = &A[(j-1)*m];
        const double * restrict an = &A[(n-j)*m];
        for (i=0; i<m; ++i) {
            wj[i].re  =  aj[i];
            wnj[i].re = -an[i];
        }
    }

    block_fft_forward(&p->fft,p->work);

    for (k=1; k<=n; ++k) {
        const cpx * restrict wk = &p->work[k*m];
        double * restrict ak = &A[(k-1)*m];
        for (i=0; i<m; ++i)
            ak[i] = scale*wk[i].im;
    }
}

/*--------------------------------------------------------------------*/
void block_fst_direct_ortho(int m, int n,
                            const double * restrict A,
                            double * restrict Y)
{
    int i,j,k;
    const double N=(double)(n+1), scale=sqrt(2.0/N);
    for (k=0; k<n; ++k) {
        for (i=0; i<m; ++i) {
            double sum=0.0;
            for (j=0; j<n; ++j)
                sum += A[i + j*m]*sin(M_PI*(double)((j+1)*(k+1))/N);
            Y[i + k*m]=scale*sum;
        }
    }
}

/*--------------------------------------------------------------------*/
int block_fst_build_matrix(block_fst_plan *p)
{
    int j,k;
    const int n = p->n;
    const double N = (double)(n+1);
    const double scale = sqrt(2.0/N);

    if (p->S) return 0;

    p->S = (double*) malloc((size_t)n*n*sizeof(double));
    if (!p->S) return 1;

    /* S[k*n+j] maps input column j to output column k. */
    for (k=0; k<n; ++k)
        for (j=0; j<n; ++j)
            p->S[k*n+j] = scale*sin(M_PI*(double)((j+1)*(k+1))/N);

    return 0;
}

/* Triple loop straight from the definition: O(m n^2). */
void block_fst_apply_slow(block_fst_plan *p, double * restrict A)
{
    int i,j,k;
    const int m=p->m, n=p->n;

    if (block_fst_build_matrix(p)) {
        fprintf(stderr,"block_fst_apply_slow: could not build S\n");
        return;
    }

    for (k=0; k<n; ++k) {
        for (i=0; i<m; ++i) {
            double sum=0.0;
            for (j=0; j<n; ++j)
                sum += p->S[k*n+j]*A[i + j*m];
            p->tmp[i + k*m]=sum;
        }
    }
    memcpy(A,p->tmp,(size_t)m*n*sizeof(double));
}

/* Same flops, but the inner loop is the unit-stride i loop: O(m n^2). */
void block_fst_apply_matmul(block_fst_plan *p, double * restrict A)
{
    int i,j,k;
    const int m=p->m, n=p->n;

    if (block_fst_build_matrix(p)) {
        fprintf(stderr,"block_fst_apply_matmul: could not build S\n");
        return;
    }

    for (i=0; i<m*n; ++i) p->tmp[i]=0.0;

    for (k=0; k<n; ++k) {
        double * restrict Yk = &p->tmp[k*m];
        for (j=0; j<n; ++j) {
            const double skj = p->S[k*n+j];
            const double * restrict Aj = &A[j*m];
            for (i=0; i<m; ++i)          /* unit stride in both operands */
                Yk[i] += skj*Aj[i];
        }
    }
    memcpy(A,p->tmp,(size_t)m*n*sizeof(double));
}
