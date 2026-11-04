#include <stdlib.h>
#include <math.h>
#include "pcg.h"

#define IDX(i, j, k) (((size_t)(i) * n + (j)) * n + (k))

/* h^2 * diagonal entry of cell (i,j,k) */
static double diag(const pcg_prob *P, int i, int j, int k)
{
    int n = P->n;
    double d = 6.0;
    d -= P->g[0] * ((i == 0) + (i == n - 1));
    d -= P->g[1] * ((j == 0) + (j == n - 1));
    d -= P->g[2] * ((k == 0) + (k == n - 1));
    return d;
}

void apply_A(const pcg_prob *P, const double *x, double *y)
{
    int n = P->n;
    double c = 1.0 / (P->h * P->h);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++) {
                size_t p = IDX(i, j, k);
                double s = diag(P, i, j, k) * x[p];
                if (i > 0)     s -= x[p - (size_t)n * n];
                if (i < n - 1) s -= x[p + (size_t)n * n];
                if (j > 0)     s -= x[p - n];
                if (j < n - 1) s -= x[p + n];
                if (k > 0)     s -= x[p - 1];
                if (k < n - 1) s -= x[p + 1];
                y[p] = c * s;
            }
}

static double dot(size_t N, const double *a, const double *b)
{
    double s = 0.0;
    for (size_t i = 0; i < N; i++) s += a[i] * b[i];
    return s;
}

/* z = M^{-1} r.  SSOR (omega = 1, symmetric Gauss-Seidel):
 * M = (D+L) D^{-1} (D+U), D = diag(A), L/U = strictly lower/upper parts of A
 * in lexicographic order. */
static void precond(pc_t pc, const pcg_prob *P, const double *r, double *z)
{
    int n = P->n;
    size_t N = (size_t)n * n * n;
    double c = 1.0 / (P->h * P->h);
    if (pc == PC_NONE) {
        for (size_t i = 0; i < N; i++) z[i] = r[i];
    } else if (pc == PC_JACOBI) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    z[IDX(i, j, k)] = r[IDX(i, j, k)] / (c * diag(P, i, j, k));
    } else {
        /* forward sweep: (D+L) w = r */
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++) {
                    size_t p = IDX(i, j, k);
                    double d = c * diag(P, i, j, k), s = r[p];
                    if (i > 0) s += c * z[p - (size_t)n * n];
                    if (j > 0) s += c * z[p - n];
                    if (k > 0) s += c * z[p - 1];
                    z[p] = s / d;                 /* w */
                }
        for (int i = 0; i < n; i++)               /* w <- D w */
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    z[IDX(i, j, k)] *= c * diag(P, i, j, k);
        /* backward sweep: (D+U) z = D w */
        for (int i = n - 1; i >= 0; i--)
            for (int j = n - 1; j >= 0; j--)
                for (int k = n - 1; k >= 0; k--) {
                    size_t p = IDX(i, j, k);
                    double d = c * diag(P, i, j, k), s = z[p];
                    if (i < n - 1) s += c * z[p + (size_t)n * n];
                    if (j < n - 1) s += c * z[p + n];
                    if (k < n - 1) s += c * z[p + 1];
                    z[p] = s / d;
                }
    }
}

pcg_info pcg_solve(const pcg_prob *P, const double *f, double *u,
                   pc_t pc, double rtol, int maxit, double *hist)
{
    size_t N = (size_t)P->n * P->n * P->n;
    double *r = malloc(N * sizeof(double)), *z = malloc(N * sizeof(double));
    double *p = malloc(N * sizeof(double)), *q = malloc(N * sizeof(double));
    pcg_info info = {0, 0.0, 0.0, 0};

    apply_A(P, u, q);
    for (size_t i = 0; i < N; i++) r[i] = f[i] - q[i];
    precond(pc, P, r, z);
    for (size_t i = 0; i < N; i++) p[i] = z[i];
    double rz = dot(N, r, z);
    info.res0 = info.res = sqrt(dot(N, r, r));
    if (hist) hist[0] = info.res;

    while (info.iters < maxit && info.res > rtol * info.res0) {
        apply_A(P, p, q);
        double alpha = rz / dot(N, p, q);
        for (size_t i = 0; i < N; i++) { u[i] += alpha * p[i]; r[i] -= alpha * q[i]; }
        precond(pc, P, r, z);
        double rz_new = dot(N, r, z), beta = rz_new / rz;
        rz = rz_new;
        for (size_t i = 0; i < N; i++) p[i] = z[i] + beta * p[i];
        info.res = sqrt(dot(N, r, r));
        info.iters++;
        if (hist) hist[info.iters] = info.res;
    }
    info.converged = info.res <= rtol * info.res0;
    free(r); free(z); free(p); free(q);
    return info;
}
