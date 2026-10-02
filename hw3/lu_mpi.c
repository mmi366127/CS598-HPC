#include <stdlib.h>
#include <string.h>
#include "lu_mpi.h"
#include <cblas.h>
#include <cblas.h>
#include "msg.h"

int lu_local_below(int g, int c, int C)
{
    return g > c ? (g - c + C - 1) / C : 0;
}

int lu_global(int li, int c, int C)
{
    return li * C + c;
}

void lu_dist_alloc(lu_dist *d, int n, int P, int Q, int rank)
{
    d->n = n; d->P = P; d->Q = Q;
    d->p = rank / Q; d->q = rank % Q;
    d->mloc = lu_local_below(n, d->p, P);
    d->nloc = lu_local_below(n, d->q, Q);
    d->a = calloc((size_t)d->mloc * d->nloc + 1, sizeof(double));
}

void lu_dist_free(lu_dist *d) { free(d->a); d->a = NULL; }

void lu_dist_fill(lu_dist *d, double (*f)(int, int))
{
    for (int li = 0; li < d->mloc; li++) {
        int gi = lu_global(li, d->p, d->P);
        for (int lj = 0; lj < d->nloc; lj++) {
            int gj = lu_global(lj, d->q, d->Q);
            d->a[(size_t)li*d->nloc + lj] = f(gi, gj);
        }
    }
}

void lu_factor_serial(double *a, int n)
{
    for (int k = 0; k < n; k++) {
        double piv = a[(size_t)k*n + k];
        for (int i = k+1; i < n; i++) {
            double l = a[(size_t)i*n + k] /= piv;
            for (int j = k+1; j < n; j++)
                a[(size_t)i*n + j] -= l * a[(size_t)k*n + j];
        }
    }
}

#define TAG_PIV 101
#define TAG_L   102
#define TAG_U   103

/*
 * Right-looking LU with the trailing update delayed and blocked: pivot steps
 * are grouped into blocks of b consecutive steps. Within a block the L columns
 * / U rows are only stored (Lb row t holds L_t for all local rows, Ub row t
 * holds U_t for all local cols); the parts of column k / row k that step k
 * needs are brought up to date with small loops, and the rest of the trailing
 * matrix, A_ij -= L_i . U_j, is applied once per block by a single dgemm
 * with inner dimension b.
 */
void lu_factor_mpi(lu_dist *d, int b)
{
    const int n = d->n, P = d->P, Q = d->Q, p = d->p, q = d->q;
    const int ld = d->nloc, ml = d->mloc;
    double *A = d->a;
    if (b < 1) b = 1;

    double *Lb = malloc(sizeof(double) * (size_t)(ml + 1) * b);   /* b x mloc */
    double *Ub = malloc(sizeof(double) * (size_t)(ld + 1) * b);   /* b x nloc */

    for (int k0 = 0; k0 < n; k0 += b) {
        const int kend = k0 + b < n ? k0 + b : n;
        for (int k = k0; k < kend; k++) {
            const int s = k - k0;
            const int pr = k % P, pc = k % Q;
            const int in_row = (p == pr), in_col = (q == pc);
            const int lk  = lu_local_below(k, p, P);     /* local row k (if in_row) */
            const int lkc = lu_local_below(k, q, Q);     /* local col k (if in_col) */
            const int lit = lu_local_below(k+1, p, P);
            const int ljt = lu_local_below(k+1, q, Q);
            const int mt = ml - lit, nt = ld - ljt;
#define MT(r) (lu_local_below(n, (r), P) - lu_local_below(k+1, (r), P))
#define NT(c) (lu_local_below(n, (c), Q) - lu_local_below(k+1, (c), Q))

            /* 0. apply the s pending updates of this block to column k
             *    (rows >= k) and row k (cols > k) */
            if (s > 0) {
                if (in_col)
                    for (int i = (in_row ? lk : lit); i < ml; i++) {
                        double acc = 0;
                        for (int t = 0; t < s; t++)
                            acc += Lb[(size_t)t*ml + i] * Ub[(size_t)t*ld + lkc];
                        A[(size_t)i*ld + lkc] -= acc;
                    }
                if (in_row && nt > 0)
                    for (int t = 0; t < s; t++) {
                        double l = Lb[(size_t)t*ml + lk];
                        const double *u = &Ub[(size_t)t*ld + ljt];
                        double *c = &A[(size_t)lk*ld + ljt];
                        for (int j = 0; j < nt; j++) c[j] -= l * u[j];
                    }
            }

            /* A. pivot goes down its process column */
            double piv = 0;
            if (in_row && in_col) {
                piv = A[(size_t)lk*ld + lkc];
                for (int r = 0; r < P; r++)
                    if (r != pr && MT(r) > 0) isend(r*Q + pc, &piv, sizeof piv, TAG_PIV);
            } else if (in_col && mt > 0) {
                irecv(pr*Q + pc, &piv, sizeof piv, TAG_PIV);
            }
            msgwait();

            /* B. L column (stored in Lb[s]); U row (stored in Ub[s]) */
            double *Ls = &Lb[(size_t)s*ml + lit];
            double *Us = &Ub[(size_t)s*ld + ljt];
            if (in_col && mt > 0)
                for (int i = 0; i < mt; i++)
                    Ls[i] = A[(size_t)(lit+i)*ld + lkc] /= piv;
            if (in_row && nt > 0)
                memcpy(Us, &A[(size_t)lk*ld + ljt], nt*sizeof(double));

            /* C. L along process rows, U along process columns */
            if (in_col && mt > 0) {
                for (int c = 0; c < Q; c++)
                    if (c != pc && NT(c) > 0) isend(p*Q + c, Ls, mt*sizeof(double), TAG_L);
            } else if (!in_col && mt > 0 && nt > 0) {
                irecv(p*Q + pc, Ls, mt*sizeof(double), TAG_L);
            }
            if (in_row && nt > 0) {
                for (int r = 0; r < P; r++)
                    if (r != pr && MT(r) > 0) isend(r*Q + q, Us, nt*sizeof(double), TAG_U);
            } else if (!in_row && mt > 0 && nt > 0) {
                irecv(pr*Q + q, Us, nt*sizeof(double), TAG_U);
            }
            msgwait();
        }

        /* D. A_ij -= L_i . U_j for rows, cols >= kend, as one dgemm:
         *    A -= L (M x bk) * U (bk x N); Lb is bk x mloc, hence Trans */
        const int bk = kend - k0;
        const int lie = lu_local_below(kend, p, P);
        const int lje = lu_local_below(kend, q, Q);
        const int M = ml - lie, N = ld - lje;
        if (M > 0 && N > 0)
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, bk,
                        -1.0, &Lb[lie], ml, &Ub[lje], ld,
                        1.0, &A[(size_t)lie*ld + lje], ld);
    }
    free(Lb); free(Ub);
}

double *lu_gather(const lu_dist *d)
{
    const int n = d->n, P = d->P, Q = d->Q;
    if (msg_rank() != 0) {
        isend(0, d->a, (size_t)d->mloc*d->nloc*sizeof(double), 900);
        msgwait();
        return NULL;
    }
    double *G = malloc(sizeof(double) * (size_t)n * n);
    for (int r = 0; r < P*Q; r++) {
        int pp = r / Q, qq = r % Q;
        int ml = lu_local_below(n, pp, P), nl = lu_local_below(n, qq, Q);
        double *buf = (r == 0) ? d->a : malloc((size_t)ml*nl*sizeof(double) + 1);
        if (r) { irecv(r, buf, (size_t)ml*nl*sizeof(double), 900); msgwait(); }
        for (int li = 0; li < ml; li++)
            for (int lj = 0; lj < nl; lj++)
                G[(size_t)lu_global(li, pp, P)*n + lu_global(lj, qq, Q)] = buf[(size_t)li*nl + lj];
        if (r) free(buf);
    }
    return G;
}
