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

/* Allgather the 2x2 Schur pieces [s11 s12 f21 s21 s22 f22], solve the
 * (P-1) reduced tridiagonal system redundantly, return this rank's
 * neighbouring separator values yl (left) and yr (right). */
static void schur_solve(const double piece[6], double *yl, double *yr,
                        tridiag_times *t)
{
    const int r = msg_rank(), P = num_ranks();
    double t0 = msg_wtime(), t1;
    *yl = *yr = 0.0;
    if (P == 1) {
        if (t) t->t_comm = t->t_reduced = 0.0;
        return;
    }
    /* allgather via sum of zero-padded slots */
    double *mine = calloc(12 * (size_t)P, sizeof(double));
    double *all  = mine + 6 * P;
    memcpy(mine + 6 * r, piece, 6 * sizeof(double));
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
    if (r > 0)      *yl = y[r - 1];
    if (r < P - 1)  *yr = y[r];
    free(mine);
    free(rd);
    if (t) t->t_reduced = msg_wtime() - t0;
}

/* l_1: b of the left neighbour's separator row (one scalar from rank r-1) */
static double fetch_left_b(const double *b, int n)
{
    const int r = msg_rank(), P = num_ranks();
    double l = 0.0, bs = (r < P - 1) ? b[n] : 0.0;
    if (r > 0)     irecv(r - 1, &l, sizeof l, 0);
    if (r < P - 1) isend(r + 1, &bs, sizeof bs, 0);
    msgwait();
    return l;
}

/*
 * Handout algorithm: eliminate the extended local matrix (4) in one forward
 * sweep. n interior rows, then (if has_right) the row of the right separator.
 *   u   fill column of the left separator (u_1 = c_1, u_{k+1} = -c_{k+1} u_k)
 *   u_n2 right coupling b_n (no fill: only the last row touches it)
 *   l   rolling entry of the left separator's row (l_{k+1} = -l_k b_k)
 *   l2n coupling c of the right separator's row to x_n
 *   s11 s12 / s21 s22  local 2x2 Schur block, f21 f22 its reduced rhs
 * Inputs are not modified; pinv, f and u of every row are kept for the
 * back substitution.
 */
void tridiag_solve_mpi(int k, const double *a, const double *b,
                       const double *c, const double *f, double *x,
                       tridiag_times *t)
{
    const int r = msg_rank(), P = num_ranks();
    const int has_left = r > 0, has_right = r < P - 1;
    const int n = k - has_right;            /* interior rows */
    double t0 = msg_wtime();

    double l = fetch_left_b(b, n);          /* l_1 */
    double *work = malloc(3 * (size_t)n * sizeof(double));
    double *pinv = work, *fk_s = pinv + n, *uk_s = fk_s + n;

    double u    = has_left ? c[0] : 0.0;    /* u_1 */
    double u_n2 = has_right ? b[n-1] : 0.0;
    double l2n  = has_right ? c[n] : 0.0;
    double s11 = 0.0, s12 = 0.0, s21 = 0.0;
    double s22  = has_right ? a[n] : 0.0;   /* a_{n+1} */
    double f21 = 0.0, f22 = has_right ? f[n] : 0.0;
    double ak = a[0], fk = f[0], p_inv, ck;

    for (int i = 0; i < n - 1; i++) {
        p_inv = 1.0 / ak;
        pinv[i] = p_inv; fk_s[i] = fk; uk_s[i] = u;
        ck = p_inv * c[i+1];                /* multiplier */
        ak = a[i+1] - ck * b[i];
        fk = f[i+1] - ck * fk_s[i];
        u  = 0.0 - ck * uk_s[i];
        l  = p_inv * l;
        s11 -= l * uk_s[i];
        f21 -= l * fk_s[i];
        l  = -l * b[i];                     /* l_{k+1} */
    }
    /* last interior row i = n-1 */
    p_inv = 1.0 / ak;
    pinv[n-1] = p_inv; fk_s[n-1] = fk; uk_s[n-1] = u;
    l    = p_inv * l;
    s11 -= l * u;
    s12 -= l * u_n2;
    f21 -= l * fk;
    l2n  = p_inv * l2n;
    s21 -= l2n * u;
    s22 -= l2n * u_n2;
    f22 -= l2n * fk;

    const double piece[6] = {s11, s12, f21, s21, s22, f22};
    double t1 = msg_wtime();
    if (t) t->t_local = t1 - t0;

    double yl, yr;
    schur_solve(piece, &yl, &yr, t);

    /* back substitution: fully local */
    t0 = msg_wtime();
    x[n-1] = (fk_s[n-1] - uk_s[n-1] * yl - u_n2 * yr) * pinv[n-1];
    for (int i = n - 2; i >= 0; i--)
        x[i] = (fk_s[i] - b[i] * x[i+1] - uk_s[i] * yl) * pinv[i];
    if (has_right) x[n] = yr;
    if (t) t->t_back = msg_wtime() - t0;
    free(work);
}
