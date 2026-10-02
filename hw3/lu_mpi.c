#include <stdlib.h>
#include <string.h>
#include "lu_mpi.h"
#include "msg.h"

#define MIN(a,b) ((a)<(b)?(a):(b))

int lu_local_below(int g, int n, int nb, int c, int C)
{
    if (g > n) g = n;
    int nblk = g / nb, cnt = 0;
    /* full blocks b < nblk with b % C == c */
    if (nblk > c) cnt = ((nblk - c - 1) / C + 1) * nb;
    if (g % nb && nblk % C == c) cnt += g % nb;
    return cnt;
}

int lu_global(int li, int nb, int c, int C)
{
    return ((li / nb) * C + c) * nb + li % nb;
}

void lu_dist_alloc(lu_dist *d, int n, int nb, int P, int Q, int rank)
{
    d->n = n; d->nb = nb; d->P = P; d->Q = Q;
    d->p = rank / Q; d->q = rank % Q;
    d->mloc = lu_local_below(n, n, nb, d->p, P);
    d->nloc = lu_local_below(n, n, nb, d->q, Q);
    d->a = calloc((size_t)d->mloc * d->nloc + 1, sizeof(double));
}

void lu_dist_free(lu_dist *d) { free(d->a); d->a = NULL; }

void lu_dist_fill(lu_dist *d, double (*f)(int, int))
{
    for (int li = 0; li < d->mloc; li++) {
        int gi = lu_global(li, d->nb, d->p, d->P);
        for (int lj = 0; lj < d->nloc; lj++)
            d->a[(size_t)li*d->nloc + lj] = f(gi, lu_global(lj, d->nb, d->q, d->Q));
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

/* Unblocked LU of a bs x bs block (leading dim ld). */
static void factor_block(double *a, int ld, int bs)
{
    for (int k = 0; k < bs; k++)
        for (int i = k+1; i < bs; i++) {
            double l = a[i*ld + k] /= a[k*ld + k];
            for (int j = k+1; j < bs; j++) a[i*ld + j] -= l * a[k*ld + j];
        }
}

#define TAG_DIAG 101
#define TAG_L    102
#define TAG_U    103

void lu_factor_mpi(lu_dist *d)
{
    const int n = d->n, nb = d->nb, P = d->P, Q = d->Q, p = d->p, q = d->q;
    const int ld = d->nloc;
    double *A = d->a;
    const int nblk = (n + nb - 1) / nb;

    double *diag = malloc(sizeof(double) * nb * nb);
    double *Lp   = malloc(sizeof(double) * (size_t)(d->mloc + 1) * nb);  /* mt x bs */
    double *Up   = malloc(sizeof(double) * (size_t)nb * (d->nloc + 1));  /* bs x nt */

    for (int k = 0; k < nblk; k++) {
        const int k0 = k * nb, bs = MIN(nb, n - k0);
        const int pr = k % P, pc = k % Q;
        const int in_row = (p == pr), in_col = (q == pc);

        /* local offsets: diagonal block, and start of trailing region */
        const int li0 = lu_local_below(k0, n, nb, p, P);
        const int lj0 = lu_local_below(k0, n, nb, q, Q);
        const int lit = lu_local_below(k0 + bs, n, nb, p, P);
        const int ljt = lu_local_below(k0 + bs, n, nb, q, Q);
        const int mt = d->mloc - lit, nt = d->nloc - ljt;
        /* trailing rows/cols owned by grid row r / grid col c (for send guards) */
#define MT(r) (lu_local_below(n, n, nb, (r), P) - lu_local_below(k0+bs, n, nb, (r), P))
#define NT(c) (lu_local_below(n, n, nb, (c), Q) - lu_local_below(k0+bs, n, nb, (c), Q))

        /* A. factor diagonal block, send along its process row and column */
        if (in_row && in_col) {
            for (int i = 0; i < bs; i++)
                memcpy(&diag[i*bs], &A[(size_t)(li0+i)*ld + lj0], bs*sizeof(double));
            factor_block(diag, bs, bs);
            for (int i = 0; i < bs; i++)
                memcpy(&A[(size_t)(li0+i)*ld + lj0], &diag[i*bs], bs*sizeof(double));
            for (int c = 0; c < Q; c++)
                if (c != pc && NT(c) > 0) isend(pr*Q + c, diag, bs*bs*sizeof(double), TAG_DIAG);
            for (int r = 0; r < P; r++)
                if (r != pr && MT(r) > 0) isend(r*Q + pc, diag, bs*bs*sizeof(double), TAG_DIAG);
        } else if (in_row && nt > 0) {
            irecv(pr*Q + pc, diag, bs*bs*sizeof(double), TAG_DIAG);
        } else if (in_col && mt > 0) {
            irecv(pr*Q + pc, diag, bs*bs*sizeof(double), TAG_DIAG);
        }
        msgwait();

        /* B. panel solves */
        if (in_col && mt > 0) {
            /* L_ik = A_ik U_kk^{-1}; pack into Lp (mt x bs) */
            for (int i = 0; i < mt; i++) {
                double *x = &A[(size_t)(lit+i)*ld + lj0];
                for (int j = 0; j < bs; j++) {
                    double s = x[j];
                    for (int t = 0; t < j; t++) s -= x[t] * diag[t*bs + j];
                    x[j] = s / diag[j*bs + j];
                }
                memcpy(&Lp[(size_t)i*bs], x, bs*sizeof(double));
            }
        }
        if (in_row && nt > 0) {
            /* U_kj = L_kk^{-1} A_kj (unit lower); pack into Up (bs x nt) */
            for (int i = 0; i < bs; i++) {
                double *x = &A[(size_t)(li0+i)*ld + ljt];
                for (int t = 0; t < i; t++) {
                    double l = diag[i*bs + t];
                    const double *y = &A[(size_t)(li0+t)*ld + ljt];
                    for (int j = 0; j < nt; j++) x[j] -= l * y[j];
                }
                memcpy(&Up[(size_t)i*nt], x, nt*sizeof(double));
            }
        }

        /* C. broadcast L panel along process rows, U panel along process columns */
        if (in_col && mt > 0) {
            for (int c = 0; c < Q; c++)
                if (c != pc && NT(c) > 0) isend(p*Q + c, Lp, (size_t)mt*bs*sizeof(double), TAG_L);
        } else if (!in_col && mt > 0 && nt > 0) {
            irecv(p*Q + pc, Lp, (size_t)mt*bs*sizeof(double), TAG_L);
        }
        if (in_row && nt > 0) {
            for (int r = 0; r < P; r++)
                if (r != pr && MT(r) > 0) isend(r*Q + q, Up, (size_t)bs*nt*sizeof(double), TAG_U);
        } else if (!in_row && mt > 0 && nt > 0) {
            irecv(pr*Q + q, Up, (size_t)bs*nt*sizeof(double), TAG_U);
        }
        msgwait();

        /* D. trailing update A_ij -= L_ik U_kj */
        for (int i = 0; i < mt; i++) {
            double *c = &A[(size_t)(lit+i)*ld + ljt];
            for (int t = 0; t < bs; t++) {
                double l = Lp[(size_t)i*bs + t];
                const double *u = &Up[(size_t)t*nt];
                for (int j = 0; j < nt; j++) c[j] -= l * u[j];
            }
        }
    }
    free(diag); free(Lp); free(Up);
}

double *lu_gather(const lu_dist *d)
{
    const int n = d->n, nb = d->nb, P = d->P, Q = d->Q;
    int me = msg_rank();
    if (me != 0) {
        isend(0, d->a, (size_t)d->mloc*d->nloc*sizeof(double), 900);
        msgwait();
        return NULL;
    }
    double *G = malloc(sizeof(double) * (size_t)n * n);
    for (int r = 0; r < P*Q; r++) {
        int pp = r / Q, qq = r % Q;
        int ml = lu_local_below(n, n, nb, pp, P), nl = lu_local_below(n, n, nb, qq, Q);
        double *buf = (r == 0) ? d->a : malloc((size_t)ml*nl*sizeof(double) + 1);
        if (r) { irecv(r, buf, (size_t)ml*nl*sizeof(double), 900); msgwait(); }
        for (int li = 0; li < ml; li++) {
            int gi = lu_global(li, nb, pp, P);
            for (int lj = 0; lj < nl; lj++)
                G[(size_t)gi*n + lu_global(lj, nb, qq, Q)] = buf[(size_t)li*nl + lj];
        }
        if (r) free(buf);
    }
    return G;
}
