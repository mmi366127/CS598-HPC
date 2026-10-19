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

static void schur_solve(const double piece[6], double *yl, double *yr,
                        tridiag_times *t)
{
    const int p = msg_rank(), P = num_ranks();
    double t0 = msg_wtime(), t1;
    *yl = *yr = 0.0;
    if (P == 1) {
        if (t) t->t_comm = t->t_reduced = 0.0;
        return;
    }
    /* allgather via sum of zero-padded slots */
    double *mine = calloc(12 * (size_t)P, sizeof(double));
    double *all  = mine + 6 * P;
    memcpy(mine + 6 * p, piece, 6 * sizeof(double));
    gsum_double(mine, all, 6 * P);
    t1 = msg_wtime();
    if (t) t->t_comm = t1 - t0;
    t0 = t1;

    /* reduced system on separators j = 0..P-2 (tridiagonal) */
    int ns = P - 1;
    double *rd = malloc(6 * (size_t)ns * sizeof(double));
    double *sa = rd, *sb = sa + ns, *sc = sb + ns, *sf = sc + ns;
    double *sw = sf + ns, *y = sw + ns;
    for (int j = 0; j < ns; j++) {
        const double *Gj = all + 6 * j, *Gn = all + 6 * (j + 1);
        sa[j] = Gj[4] + Gn[0];
        sc[j] = Gj[3];              /* coeff of y_{j-1} */
        sb[j] = Gn[1];              /* coeff of y_{j+1} */
        sf[j] = Gj[5] + Gn[2];
    }
    tridiag_thomas(ns, sa, sb, sc, sf, y, sw);
    if (p > 0)      *yl = y[p - 1];
    if (p < P - 1)  *yr = y[p];
    free(mine);
    free(rd);
    if (t) t->t_reduced = msg_wtime() - t0;
}

void tridiag_solve_mpi(int k, const double *a, const double *b,
                       const double *c, const double *f, double *x,
                       tridiag_times *t)
{
    const int p = msg_rank(), P = num_ranks();
    const int has_left = p > 0, has_right = p < P - 1;
    const int n = k - has_right;            /* interior rows */
    double t0 = msg_wtime();

    /* l_1: b of the left neighbour's separator row (one scalar from rank p-1) */
    double l = 0.0, bs = has_right ? b[n] : 0.0;
    if (has_left)  irecv(p - 1, &l, sizeof l, 0);
    if (has_right) isend(p + 1, &bs, sizeof bs, 0);
    msgwait();

    /* work buffer is kept between calls: re-allocating (and page-faulting)
     * 3n doubles on every solve would dominate the timing for large n */
    static double *work = NULL;
    static size_t work_cap = 0;
    if ((size_t)n > work_cap) {
        free(work);
        work = malloc(3 * (size_t)n * sizeof(double));
        work_cap = (size_t)n;
    }
    double *pinv = work, *fk_s = pinv + n, *uk_s = fk_s + n;

    double u    = has_left  ? c[0] : 0.0;
    double un2 = has_right ? b[n-1] : 0.0;
    double l2n  = has_right ? c[n] : 0.0;
    double s11 = 0.0;
    double s12 = 0.0;
    double s21 = 0.0;
    double s22  = has_right ? a[n] : 0.0;
    double fs1 = 0.0;
    double fs2 = has_right ? f[n] : 0.0;
    double ak = a[0];
    double fk = f[0];
    double ck = 0.0;

    for (int i = 0; i < n - 1; i++) {
        pinv[i] = 1.0 / ak;
        fk_s[i] = fk;
        uk_s[i] = u;
        ck = pinv[i] * c[i+1];
        ak = a[i+1] - ck * b[i];
        fk = f[i+1] - ck * fk_s[i];
        u  = 0.0 - ck * uk_s[i];
        l  = pinv[i] * l;
        s11 -= l * uk_s[i];
        fs1 -= l * fk_s[i];
        l  = -l * b[i];
    }
    /* last interior row i = n-1 */
    pinv[n-1] = 1.0 / ak;
    fk_s[n-1] = fk;
    uk_s[n-1] = u;
    l    = pinv[n-1] * l;
    s11 -= l * u;
    s12 -= l * un2;
    fs1 -= l * fk;
    l2n  = pinv[n-1] * l2n;
    s21 -= l2n * u;
    s22 -= l2n * un2;
    fs2 -= l2n * fk;

    const double piece[6] = {s11, s12, fs1, s21, s22, fs2};
    double t1 = msg_wtime();
    if (t) t->t_local = t1 - t0;

    double yl, yr;
    schur_solve(piece, &yl, &yr, t);

    /* back substitution: fully local */
    t0 = msg_wtime();
    x[n-1] = (fk_s[n-1] - uk_s[n-1] * yl - un2 * yr) * pinv[n-1];
    for (int i = n - 2; i >= 0; i--)
        x[i] = (fk_s[i] - b[i] * x[i+1] - uk_s[i] * yl) * pinv[i];
    if (has_right) x[n] = yr;
    if (t) t->t_back = msg_wtime() - t0;
}
