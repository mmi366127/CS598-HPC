#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "block_fft.h"
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static int compute_stages(int n, int radix)
{
    int stages = 0, m = 1;
    while (m < n) { m *= radix; stages++; }
    return (m == n) ? stages : -1;
}

static int digit_reverse_base_r(int i, int radix, int stages)
{
    int j = 0, s;
    for (s = 0; s < stages; ++s) {
        int d = i % radix;
        i /= radix;
        j = radix*j + d;
    }
    return j;
}

int block_fft_plan_init(block_fft_plan *p, int n, int q, int radix)
{
    int i, len, m, j, off, ntw;
    p->n=n; p->q=q; p->radix=radix; p->stages=compute_stages(n,radix);
    p->rev=0; p->W1=0; p->W2=0; p->W3=0;

    if (radix != 4) { fprintf(stderr,"block_fft: radix must be 4\n"); return 1; }
    if (p->stages < 0) { fprintf(stderr,"block_fft: require n=4^k\n"); return 2; }

    p->rev = (int*) malloc((size_t)n*sizeof(int));
    ntw = (n-1)/3;
    p->W1 = (cpx*) malloc((size_t)ntw*sizeof(cpx));
    p->W2 = (cpx*) malloc((size_t)ntw*sizeof(cpx));
    p->W3 = (cpx*) malloc((size_t)ntw*sizeof(cpx));
    if (!p->rev || !p->W1 || !p->W2 || !p->W3) {
        block_fft_plan_free(p); return 3;
    }

    for (i=0; i<n; ++i) p->rev[i] = digit_reverse_base_r(i,radix,p->stages);

    off = 0;
    for (len=4; len<=n; len*=4) {
        m = len/4;
        for (j=0; j<m; ++j) {
            double th1 = -2.0*M_PI*(double)j/(double)len;
            double th2 = 2.0*th1, th3 = 3.0*th1;
            p->W1[off+j].re=cos(th1); p->W1[off+j].im=sin(th1);
            p->W2[off+j].re=cos(th2); p->W2[off+j].im=sin(th2);
            p->W3[off+j].re=cos(th3); p->W3[off+j].im=sin(th3);
        }
        off += m;
    }
    return 0;
}

void block_fft_plan_free(block_fft_plan *p)
{
    free(p->rev); free(p->W1); free(p->W2); free(p->W3);
    p->rev=0; p->W1=0; p->W2=0; p->W3=0;
}

static void block_digit_reverse(block_fft_plan *p, cpx * restrict x)
{
    int i,r, n=p->n, q=p->q;
    for (i=0; i<n; ++i) {
        int j = p->rev[i];
        if (i < j) {
            cpx * restrict xi=&x[i*q], * restrict xj=&x[j*q];
            for (r=0; r<q; ++r) {
                cpx tmp=xi[r]; xi[r]=xj[r]; xj[r]=tmp;
            }
        }
    }
}

void block_fft_forward(block_fft_plan *p, cpx * restrict x)
{
    int len,m,block,j,r,off=0;
    const int n=p->n, q=p->q;

    block_digit_reverse(p,x);

    for (len=4; len<=n; len*=4) {
        m = len/4;
        for (block=0; block<n; block+=len) {
            for (j=0; j<m; ++j) {
                cpx * restrict x0=&x[(block+j+0*m)*q];
                cpx * restrict x1=&x[(block+j+1*m)*q];
                cpx * restrict x2=&x[(block+j+2*m)*q];
                cpx * restrict x3=&x[(block+j+3*m)*q];

                const double w1r=p->W1[off+j].re, w1i=p->W1[off+j].im;
                const double w2r=p->W2[off+j].re, w2i=p->W2[off+j].im;
                const double w3r=p->W3[off+j].re, w3i=p->W3[off+j].im;

                for (r=0; r<q; ++r) {
                    const double a0r=x0[r].re, a0i=x0[r].im;

                    const double b1r=x1[r].re, b1i=x1[r].im;
                    const double a1r=w1r*b1r - w1i*b1i;
                    const double a1i=w1r*b1i + w1i*b1r;

                    const double b2r=x2[r].re, b2i=x2[r].im;
                    const double a2r=w2r*b2r - w2i*b2i;
                    const double a2i=w2r*b2i + w2i*b2r;

                    const double b3r=x3[r].re, b3i=x3[r].im;
                    const double a3r=w3r*b3r - w3i*b3i;
                    const double a3i=w3r*b3i + w3i*b3r;

                    x0[r].re = a0r + a1r + a2r + a3r;
                    x0[r].im = a0i + a1i + a2i + a3i;

                    x1[r].re = a0r + a1i - a2r - a3i;
                    x1[r].im = a0i - a1r - a2i + a3r;

                    x2[r].re = a0r - a1r + a2r - a3r;
                    x2[r].im = a0i - a1i + a2i - a3i;

                    x3[r].re = a0r - a1i - a2r + a3i;
                    x3[r].im = a0i + a1r - a2i - a3r;
                }
            }
        }
        off += m;
    }
}
