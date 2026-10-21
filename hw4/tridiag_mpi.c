#include <stdlib.h>
#include <string.h>
#include "msg.h"
#include "tridiag_mpi.h"

static double *pinv = NULL;
static double *uk_s = NULL;
static double *lk_s = NULL;
static double l2n = 0.0;
static double un2 = 0.0;
static size_t work_cap = 0;

// Schur part
static double *red = NULL;
static double *red_sb = NULL, *red_lm = NULL, *red_sw = NULL;

static void schur_factor(const double piece[4], tridiag_times *t)
{
    const int P = num_ranks();
    double t0 = msg_wtime(), t1;
    free(red);
    red = NULL;
    if (P == 1) {
        if (t) t->t_comm = t->t_reduced = 0.0;
        return;
    }
    /* allgather via sum of zero-padded slots */
    double *mine = calloc(8 * (size_t)P, sizeof(double));
    double *all  = mine + 4 * P;
    memcpy(mine + 4 * msg_rank(), piece, 4 * sizeof(double));
    gsum_double(mine, all, 4 * P);
    t1 = msg_wtime();
    if (t) t->t_comm = t1 - t0;
    t0 = t1;

    int ns = P - 1;
    red = malloc(3 * (size_t)ns * sizeof(double));
    red_sb = red; red_lm = red_sb + ns; red_sw = red_lm + ns;
    for (int j = 0; j < ns; j++) {
        const double *Gj = all + 4 * j, *Gn = all + 4 * (j + 1);
        red_sb[j] = Gn[1];                  /* coeff of y_{j+1} */
        red_sw[j] = Gj[3] + Gn[0];
        if (j > 0) {
            red_lm[j] = Gj[2] / red_sw[j-1];    /* Gj[2]: coeff of y_{j-1} */
            red_sw[j] -= red_lm[j] * red_sb[j-1];
        }
    }
    free(mine);
    if (t) t->t_reduced = msg_wtime() - t0;
}

static void schur_substitute(const double g[2], double *yl, double *yr,
                            tridiag_times *t)
{
    const int p = msg_rank(), P = num_ranks();
    double t0 = msg_wtime(), t1;
    *yl = *yr = 0.0;
    if (P == 1) {
        if (t) t->t_comm = t->t_reduced = 0.0;
        return;
    }
    double *mine = calloc(4 * (size_t)P, sizeof(double));
    double *all  = mine + 2 * P;
    memcpy(mine + 2 * p, g, 2 * sizeof(double));
    gsum_double(mine, all, 2 * P);
    t1 = msg_wtime();
    if (t) t->t_comm = t1 - t0;
    t0 = t1;

    int ns = P - 1;
    double *y = malloc((size_t)ns * sizeof(double));
    y[0] = all[1] + all[2];
    for (int i = 1; i < ns; i++)
        y[i] = all[2 * i + 1] + all[2 * i + 2] - red_lm[i] * y[i-1];
    y[ns-1] /= red_sw[ns-1];
    for (int i = ns - 2; i >= 0; i--)
        y[i] = (y[i] - red_sb[i] * y[i+1]) / red_sw[i];
    if (p > 0)      *yl = y[p - 1];
    if (p < P - 1)  *yr = y[p];
    free(mine);
    free(y);
    if (t) t->t_reduced = msg_wtime() - t0;
}

void tridiag_factor_mpi(int k, const double *a, const double *b,
                        const double *c, tridiag_times *t)
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


    // Storage alloc
    if ((size_t)n > work_cap) {
        free(pinv);
        free(uk_s);
        free(lk_s);
        pinv = malloc((size_t)n * sizeof(double));
        uk_s = malloc((size_t)n * sizeof(double));
        lk_s = malloc((size_t)n * sizeof(double));
        work_cap = (size_t)n;
    }

    // Main loop
    double u    = has_left  ? c[0] : 0.0;
    double s11 = 0.0;
    double s12 = 0.0;
    double s21 = 0.0;
    double s22  = has_right ? a[n] : 0.0;
    double ak = a[0];
    double ck = 0.0;

    for (int i = 0; i < n - 1; i++) {
        pinv[i] = 1.0 / ak;
        uk_s[i] = u;
        ck = pinv[i] * c[i+1];
        ak = a[i+1] - ck * b[i];
        u  = 0.0 - ck * uk_s[i];
        l  = pinv[i] * l;
        lk_s[i] = l;
        s11 -= l * uk_s[i];
        l  = -l * b[i];
    }

    // Last row
    un2 = has_right ? b[n-1] : 0.0;
    l2n  = has_right ? c[n] : 0.0;
    pinv[n-1] = 1.0 / ak;
    uk_s[n-1] = u;
    l    = pinv[n-1] * l;
    lk_s[n-1] = l;
    s11 -= l * u;
    s12 -= l * un2;
    l2n  = pinv[n-1] * l2n;
    s21 -= l2n * u;
    s22 -= l2n * un2;

    const double piece[4] = {s11, s12, s21, s22};
    double t1 = msg_wtime();
    if (t) t->t_local = t1 - t0;

    // Schur
    schur_factor(piece, t);
}

void tridiag_substitute_mpi(int k, const double *b, const double *c,
                           const double *f, double *x, tridiag_times *t)
{
    const int p = msg_rank(), P = num_ranks();
    const int has_right = p < P - 1;
    const int n = k - has_right;
    double t0 = msg_wtime();

    /* forward elimination of the RHS; fk_s: transformed RHS of the interior rows */
    double *fk_s = malloc((size_t)n * sizeof(double));
    double fs1 = 0.0;
    double fs2 = has_right ? f[n] : 0.0;
    double fk = f[0];
    for (int i = 0; i < n - 1; i++) {
        fk_s[i] = fk;
        fk = f[i+1] - pinv[i] * c[i+1] * fk_s[i];
        fs1 -= lk_s[i] * fk_s[i];
    }
    fk_s[n-1] = fk;
    fs1 -= lk_s[n-1] * fk;
    fs2 -= l2n * fk;

    const double g[2] = {fs1, fs2};
    double t1 = msg_wtime();
    if (t) t->t_local = t1 - t0;

    double yl, yr;
    schur_substitute(g, &yl, &yr, t);

    /* back substitution: fully local */
    t0 = msg_wtime();
    x[n-1] = (fk_s[n-1] - uk_s[n-1] * yl - un2 * yr) * pinv[n-1];
    for (int i = n - 2; i >= 0; i--)
        x[i] = (fk_s[i] - b[i] * x[i+1] - uk_s[i] * yl) * pinv[i];
    if (has_right) x[n] = yr;
    free(fk_s);
    if (t) t->t_back = msg_wtime() - t0;
}
