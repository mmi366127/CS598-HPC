#include <stdlib.h>
#include <string.h>
#include "lu_mpi.h"
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
        for (int lj = 0; lj < d->nloc; lj++)
            d->a[(size_t)li*d->nloc + lj] = f(gi, lu_global(lj, d->q, d->Q));
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

void lu_factor_mpi(lu_dist *d)
{
    const int n = d->n, P = d->P, Q = d->Q, p = d->p, q = d->q;
    const int ld = d->nloc;
    double *A = d->a;

    double *Lp = malloc(sizeof(double) * (d->mloc + 1));  /* mt */
    double *Up = malloc(sizeof(double) * (d->nloc + 1));  /* nt */

    for (int k = 0; k < n - 1; k++) {
        const int pr = k % P, pc = k % Q;
        const int in_row = (p == pr), in_col = (q == pc);

        /* local position of row/col k (if owned) and of the trailing region */
        const int lk  = lu_local_below(k, p, P);
        const int lkc = lu_local_below(k, q, Q);
        const int lit = lu_local_below(k+1, p, P);
        const int ljt = lu_local_below(k+1, q, Q);
        const int mt = d->mloc - lit, nt = d->nloc - ljt;
        /* trailing rows/cols owned by grid row r / grid col c (for send guards) */
#define MT(r) (lu_local_below(n, (r), P) - lu_local_below(k+1, (r), P))
#define NT(c) (lu_local_below(n, (c), Q) - lu_local_below(k+1, (c), Q))

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

        /* B. L column = A(:,k)/piv; U row = A(k,:) is already final (unit L) */
        if (in_col && mt > 0)
            for (int i = 0; i < mt; i++)
                Lp[i] = A[(size_t)(lit+i)*ld + lkc] /= piv;
        if (in_row && nt > 0)
            memcpy(Up, &A[(size_t)lk*ld + ljt], nt*sizeof(double));

        /* C. L along process rows, U along process columns */
        if (in_col && mt > 0) {
            for (int c = 0; c < Q; c++)
                if (c != pc && NT(c) > 0) isend(p*Q + c, Lp, mt*sizeof(double), TAG_L);
        } else if (!in_col && mt > 0 && nt > 0) {
            irecv(p*Q + pc, Lp, mt*sizeof(double), TAG_L);
        }
        if (in_row && nt > 0) {
            for (int r = 0; r < P; r++)
                if (r != pr && MT(r) > 0) isend(r*Q + q, Up, nt*sizeof(double), TAG_U);
        } else if (!in_row && mt > 0 && nt > 0) {
            irecv(pr*Q + q, Up, nt*sizeof(double), TAG_U);
        }
        msgwait();

        /* D. rank-1 trailing update */
        for (int i = 0; i < mt; i++) {
            double *c = &A[(size_t)(lit+i)*ld + ljt];
            double l = Lp[i];
            for (int j = 0; j < nt; j++) c[j] -= l * Up[j];
        }
    }
    free(Lp); free(Up);
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
