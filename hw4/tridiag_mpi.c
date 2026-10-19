#include <stdlib.h>
#include <string.h>
#include "msg.h"
#include "tridiag_mpi.h"

void tridiag_thomas(int n, const double *a, const double *b, const double *c,
                    const double *f, double *x, double *d)
{
    /* d: modified diagonal; x doubles as the forward-sweep vector */
    d[0] = a[0];
    x[0] = f[0];
    for (int i = 1; i < n; i++) {
        double l = c[i] / d[i-1];
        d[i] = a[i] - l * b[i-1];
        x[i] = f[i] - l * x[i-1];
    }
    x[n-1] /= d[n-1];
    for (int i = n - 2; i >= 0; i--)
        x[i] = (x[i] - b[i] * x[i+1]) / d[i];
}

void tridiag_solve_mpi(int k, const double *a, const double *b,
                       const double *c, const double *f, double *x,
                       tridiag_times *t)
{
    const int r = msg_rank(), P = num_ranks();
    const int has_left = r > 0, has_right = r < P - 1;
    const int m = k - has_right;            /* interior rows */
    double t0 = msg_wtime(), t1;

    /* one scalar from the left neighbour: l_1 = b of separator s_{r-1} */
    double l1 = 0.0, bs_send = has_right ? b[m] : 0.0;
    if (has_left)  irecv(r - 1, &l1, sizeof l1, 0);
    if (has_right) isend(r + 1, &bs_send, sizeof bs_send, 0);
    msgwait();

    double *work = malloc(5 * (size_t)m * sizeof(double));
    double *d = work, *l = d + m, *g = l + m, *v = g + m, *w = v + m;

    /* factor A_p = L U (Thomas) */
    d[0] = a[0];
    l[0] = 0.0;
    for (int i = 1; i < m; i++) {
        l[i] = c[i] / d[i-1];
        d[i] = a[i] - l[i] * b[i-1];
    }

    /* forward sweeps: g with rhs f, v with rhs c[0] e_0 (fills every row);
     * w has rhs b[m-1] e_{m-1}, zero until the last row, so it needs none */
    g[0] = f[0];
    v[0] = has_left ? c[0] : 0.0;
    for (int i = 1; i < m; i++) {
        g[i] = f[i] - l[i] * g[i-1];
        v[i] = -l[i] * v[i-1];
    }
    /* backward sweeps; w_{m-1} = b[m-1]/d, then w_i = -b_i w_{i+1}/d_i */
    g[m-1] /= d[m-1];
    v[m-1] /= d[m-1];
    w[m-1] = has_right ? b[m-1] / d[m-1] : 0.0;
    for (int i = m - 2; i >= 0; i--) {
        g[i] = (g[i] - b[i] * g[i+1]) / d[i];
        v[i] = (v[i] - b[i] * v[i+1]) / d[i];
        w[i] = -b[i] * w[i+1] / d[i];
    }

    /* local 2x2 Schur piece + reduced rhs: [S11 S12 r1 S21 S22 r2] */
    double piece[6] = {0, 0, 0, 0, 0, 0};
    if (has_left) {
        piece[0] = -l1 * v[0];
        piece[1] = -l1 * w[0];
        piece[2] = -l1 * g[0];
    }
    if (has_right) {
        double cs = c[m], as = a[m], fs = f[m];
        piece[3] = -cs * v[m-1];
        piece[4] = as - cs * w[m-1];
        piece[5] = fs - cs * g[m-1];
    }
    t1 = msg_wtime();
    if (t) t->t_local = t1 - t0;
    t0 = t1;

    /* allgather via sum of zero-padded slots */
    double yr = 0.0, yl = 0.0;
    double *y = NULL, *rd = NULL;
    if (P > 1) {
        double *mine = calloc(12 * (size_t)P, sizeof(double));
        double *all  = mine + 6 * P;
        memcpy(mine + 6 * r, piece, sizeof piece);
        gsum_double(mine, all, 6 * P);
        t1 = msg_wtime();
        if (t) t->t_comm = t1 - t0;
        t0 = t1;

        /* reduced system on separators j = 0..P-2 (tridiagonal) */
        int ns = P - 1;
        rd = malloc(6 * (size_t)ns * sizeof(double));
        double *sa = rd, *sb = sa + ns, *sc = sb + ns, *sf = sc + ns;
        double *sw = sf + ns; y = sw + ns;
        for (int j = 0; j < ns; j++) {
            const double *Gj = all + 6 * j, *Gn = all + 6 * (j + 1);
            sa[j] = Gj[4] + Gn[0];
            sc[j] = Gj[3];              /* coeff of y_{j-1} */
            sb[j] = Gn[1];              /* coeff of y_{j+1} */
            sf[j] = Gj[5] + Gn[2];
        }
        tridiag_thomas(ns, sa, sb, sc, sf, y, sw);
        free(mine);
        t1 = msg_wtime();
        if (t) t->t_reduced = t1 - t0;
        t0 = t1;
        if (has_left)  yl = y[r - 1];
        if (has_right) yr = y[r];
    } else if (t) {
        t->t_comm = t->t_reduced = 0.0;
    }

    /* back substitution: fully local */
    for (int i = 0; i < m; i++)
        x[i] = g[i] - v[i] * yl - w[i] * yr;
    if (has_right) x[m] = yr;
    if (t) t->t_back = msg_wtime() - t0;

    free(rd);
    free(work);
}
