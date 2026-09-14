#include "transpose.h"

/* Unit-stride READ of A, strided WRITE of B. */
void transpose_naive(int m, int n, const double * restrict A,
                                   double * restrict B)
{
    int i,j;
    for (j=0; j<n; ++j)
        for (i=0; i<m; ++i)
            B[j + i*n] = A[i + j*m];
}

/* Tiled: both streams move in short unit-stride runs. */
void transpose_blocked(int m, int n, const double * restrict A,
                                     double * restrict B)
{
    const int TS = TRANSPOSE_TS;
    int ii,jj,i,j;

    for (jj=0; jj<n; jj+=TS) {
        const int jmax = (jj+TS < n) ? jj+TS : n;
        for (ii=0; ii<m; ii+=TS) {
            const int imax = (ii+TS < m) ? ii+TS : m;
            for (j=jj; j<jmax; ++j)
                for (i=ii; i<imax; ++i)
                    B[j + i*n] = A[i + j*m];
        }
    }
}

void transpose_real(int m, int n, const double * restrict A,
                                  double * restrict B)
{
    transpose_blocked(m,n,A,B);
}
