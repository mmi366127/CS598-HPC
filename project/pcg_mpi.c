#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "msg.h"
#include "pcg_mpi.h"

/* ---------- decomposition ---------- */

void pcg_choose_grid(int n, int nranks, int dims[3])
{
    double best = 1e300;
    dims[0] = nranks; dims[1] = dims[2] = 1;
    for (int a = 1; a <= nranks; a++) {
        if (nranks % a) continue;
        for (int b = 1; b <= nranks / a; b++) {
            if ((nranks / a) % b) continue;
            int c = nranks / a / b;
            double s = (double)n * n * (1.0 / ((double)b * c) + 1.0 / ((double)a * c) + 1.0 / ((double)a * b));
            /* tie-break: prefer larger px >= py >= pz (k is the contiguous direction) */
            if (s < best - 1e-12 || (fabs(s - best) <= 1e-12 && a >= b && b >= c)) {
                best = s; dims[0] = a; dims[1] = b; dims[2] = c;
            }
        }
    }
}

void pcg_dist_init(pcg_dist *D, const pcg_prob *P, const int dims[3])
{
    int r = msg_rank();
    D->P = *P;
    for (int d = 0; d < 3; d++) D->dims[d] = dims[d];
    D->coord[0] = r / (dims[1] * dims[2]);
    D->coord[1] = (r / dims[2]) % dims[1];
    D->coord[2] = r % dims[2];
    D->N = 1;
    for (int d = 0; d < 3; d++) {
        int base = P->n / dims[d], rem = P->n % dims[d], c = D->coord[d];
        D->nl[d]  = base + (c < rem);
        D->off[d] = c * base + (c < rem ? c : rem);
        D->lo[d]  = c == 0;
        D->hi[d]  = c == dims[d] - 1;
        D->N *= (size_t)D->nl[d];
    }
    for (int f = 0; f < 6; f++) {
        int d = f / 2, c[3] = { D->coord[0], D->coord[1], D->coord[2] };
        c[d] += (f % 2) ? 1 : -1;
        D->nbr[f] = (c[d] < 0 || c[d] >= dims[d]) ? -1 : (c[0] * dims[1] + c[1]) * dims[2] + c[2];
        int a = (d + 1) % 3, b = (d + 2) % 3;
        D->fcount[f] = D->nl[a] * D->nl[b];
        D->sbuf[f] = D->nbr[f] >= 0 ? malloc(D->fcount[f] * sizeof(double)) : NULL;
        D->rbuf[f] = D->nbr[f] >= 0 ? malloc(D->fcount[f] * sizeof(double)) : NULL;
    }
}

void pcg_dist_free(pcg_dist *D)
{
    for (int f = 0; f < 6; f++) { free(D->sbuf[f]); free(D->rbuf[f]); }
}

/* ---------- padded search direction and halo exchange ---------- */

/* padded index of local cell (i,j,k); i,j,k may be -1 or nl (ghost) */
#define PIDX(D, i, j, k) ((((size_t)(i) + 1) * ((D)->nl[1] + 2) + (size_t)(j) + 1) * ((D)->nl[2] + 2) + (size_t)(k) + 1)
#define CIDX(D, i, j, k) (((size_t)(i) * (D)->nl[1] + (j)) * (D)->nl[2] + (k))

/* visit the cells of the layer at index `layer` along direction d, in a fixed order:
 * fn(p, buf) packs (copy_out=1) or unpacks (copy_out=0) */
static void face_copy(const pcg_dist *D, double *p, int d, int layer, double *buf, int pack)
{
    int a = (d + 1) % 3, b = (d + 2) % 3, m = 0;
    int idx[3];
    idx[d] = layer;
    for (int u = 0; u < D->nl[a]; u++) {
        idx[a] = u;
        for (int v = 0; v < D->nl[b]; v++) {
            idx[b] = v;
            double *q = &p[PIDX(D, idx[0], idx[1], idx[2])];
            if (pack) buf[m++] = *q; else *q = buf[m++];
        }
    }
}

/* physical-boundary ghosts: ghost = g[d] * adjacent interior cell */
static void face_physical(const pcg_dist *D, double *p, int d, int side)
{
    int a = (d + 1) % 3, b = (d + 2) % 3, idx[3], gidx[3];
    int in = side ? D->nl[d] - 1 : 0, gh = side ? D->nl[d] : -1;
    double g = D->P.g[d];
    for (int u = 0; u < D->nl[a]; u++)
        for (int v = 0; v < D->nl[b]; v++) {
            idx[a] = gidx[a] = u; idx[b] = gidx[b] = v; idx[d] = in; gidx[d] = gh;
            p[PIDX(D, gidx[0], gidx[1], gidx[2])] = g * p[PIDX(D, idx[0], idx[1], idx[2])];
        }
}

static void halo(const pcg_dist *D, double *p)
{
    for (int f = 0; f < 6; f++)
        if (D->nbr[f] >= 0) irecv(D->nbr[f], D->rbuf[f], D->fcount[f] * (int)sizeof(double), f ^ 1);
    for (int f = 0; f < 6; f++)
        if (D->nbr[f] >= 0) {
            int d = f / 2, side = f % 2;
            face_copy(D, p, d, side ? D->nl[d] - 1 : 0, D->sbuf[f], 1);
            isend(D->nbr[f], D->sbuf[f], D->fcount[f] * (int)sizeof(double), f);
        } else
            face_physical(D, p, f / 2, f % 2);
    msgwait();
    for (int f = 0; f < 6; f++)
        if (D->nbr[f] >= 0) {
            int d = f / 2, side = f % 2;
            face_copy(D, p, d, side ? D->nl[d] : -1, D->rbuf[f], 0);
        }
}

/* q = A p (p padded, ghosts already filled); returns local p.q */
static double stencil(const pcg_dist *D, const double *p, double *q)
{
    double c = 1.0 / (D->P.h * D->P.h), s = 0.0;
    ptrdiff_t sx = (ptrdiff_t)(D->nl[1] + 2) * (D->nl[2] + 2), sy = D->nl[2] + 2;
    for (int i = 0; i < D->nl[0]; i++)
        for (int j = 0; j < D->nl[1]; j++) {
            const double *pp = &p[PIDX(D, i, j, 0)];
            double *qq = &q[CIDX(D, i, j, 0)];
            for (int k = 0; k < D->nl[2]; k++) {
                double v = c * (6.0 * pp[k] - pp[k - sx] - pp[k + sx] - pp[k - sy] - pp[k + sy]
                                - pp[k - 1] - pp[k + 1]);
                qq[k] = v;
                s += pp[k] * v;
            }
        }
    return s;
}

/* ---------- preconditioners ---------- */

/* h^2 * diagonal of local cell (i,j,k) */
static inline double dloc(const pcg_dist *D, int i, int j, int k)
{
    return 6.0 - D->P.g[0] * ((D->lo[0] && i == 0) + (D->hi[0] && i == D->nl[0] - 1))
               - D->P.g[1] * ((D->lo[1] && j == 0) + (D->hi[1] && j == D->nl[1] - 1))
               - D->P.g[2] * ((D->lo[2] && k == 0) + (D->hi[2] && k == D->nl[2] - 1));
}

static void precond(const pcg_dist *D, pc_t pc, const double *r, double *z)
{
    int n0 = D->nl[0], n1 = D->nl[1], n2 = D->nl[2];
    double c = 1.0 / (D->P.h * D->P.h);
    size_t s0 = (size_t)n1 * n2, s1 = n2;
    if (pc == PC_JACOBI) {
        for (int i = 0; i < n0; i++)
            for (int j = 0; j < n1; j++)
                for (int k = 0; k < n2; k++)
                    z[CIDX(D, i, j, k)] = r[CIDX(D, i, j, k)] / (c * dloc(D, i, j, k));
        return;
    }
    /* SSOR inside the block: forward (D+L) w = r, scale by D, backward (D+U) z = D w */
    for (int i = 0; i < n0; i++)
        for (int j = 0; j < n1; j++)
            for (int k = 0; k < n2; k++) {
                size_t p = CIDX(D, i, j, k);
                double s = r[p];
                if (i > 0) s += c * z[p - s0];
                if (j > 0) s += c * z[p - s1];
                if (k > 0) s += c * z[p - 1];
                z[p] = s / (c * dloc(D, i, j, k));
            }
    for (int i = 0; i < n0; i++)
        for (int j = 0; j < n1; j++)
            for (int k = 0; k < n2; k++)
                z[CIDX(D, i, j, k)] *= c * dloc(D, i, j, k);
    for (int i = n0 - 1; i >= 0; i--)
        for (int j = n1 - 1; j >= 0; j--)
            for (int k = n2 - 1; k >= 0; k--) {
                size_t p = CIDX(D, i, j, k);
                double s = z[p];
                if (i < n0 - 1) s += c * z[p + s0];
                if (j < n1 - 1) s += c * z[p + s1];
                if (k < n2 - 1) s += c * z[p + 1];
                z[p] = s / (c * dloc(D, i, j, k));
            }
}

/* ---------- solver ---------- */

pcg_info pcg_dist_solve(const pcg_dist *D, const double *f, double *u, pc_t pc,
                        double rtol, int maxit, double *hist, pcg_times *tm)
{
    size_t N = D->N, NP = (size_t)(D->nl[0] + 2) * (D->nl[1] + 2) * (D->nl[2] + 2);
    double *r = malloc(N * sizeof(double)), *q = malloc(N * sizeof(double));
    double *zb = pc == PC_NONE ? NULL : malloc(N * sizeof(double));
    double *z = pc == PC_NONE ? r : zb;
    double *p = calloc(NP, sizeof(double));
    pcg_times T = {0, 0, 0, 0, 0, 0};
    pcg_info info = {0, 0.0, 0.0, 0};
    double t0, t1, red[2], loc[2];
    double tstart = msg_wtime();

#define TICK(acc) do { t1 = msg_wtime(); T.acc += t1 - t0; t0 = t1; } while (0)
#define COPY_TO_P(src) \
    for (int i = 0; i < D->nl[0]; i++) for (int j = 0; j < D->nl[1]; j++) { \
        double *pp = &p[PIDX(D, i, j, 0)]; const double *ss = &(src)[CIDX(D, i, j, 0)]; \
        for (int k = 0; k < D->nl[2]; k++) pp[k] = ss[k]; }
#define UPDATE_P(beta) \
    for (int i = 0; i < D->nl[0]; i++) for (int j = 0; j < D->nl[1]; j++) { \
        double *pp = &p[PIDX(D, i, j, 0)]; const double *zz = &z[CIDX(D, i, j, 0)]; \
        for (int k = 0; k < D->nl[2]; k++) pp[k] = zz[k] + (beta) * pp[k]; }

    /* r0 = f - A u */
    t0 = msg_wtime();
    COPY_TO_P(u);                                   TICK(vec);
    halo(D, p);                                     TICK(halo);
    stencil(D, p, q);                               TICK(apply);
    for (size_t i = 0; i < N; i++) r[i] = f[i] - q[i];
    TICK(vec);
    if (pc != PC_NONE) { precond(D, pc, r, z); TICK(prec); }
    loc[0] = loc[1] = 0.0;
    for (size_t i = 0; i < N; i++) { loc[0] += r[i] * z[i]; loc[1] += r[i] * r[i]; }
    COPY_TO_P(z);                                   TICK(vec);
    gsum_double(loc, red, 2);                       TICK(reduce);
    double rz = red[0];
    info.res0 = info.res = sqrt(red[1]);
    if (hist) hist[0] = info.res;

    while (info.iters < maxit && info.res > rtol * info.res0) {
        t0 = msg_wtime();
        halo(D, p);                                 TICK(halo);
        double pq = stencil(D, p, q);               TICK(apply);
        gsum_double(&pq, &red[0], 1);               TICK(reduce);
        double alpha = rz / red[0], rr = 0.0;
        for (int i = 0; i < D->nl[0]; i++)
            for (int j = 0; j < D->nl[1]; j++) {
                const double *pp = &p[PIDX(D, i, j, 0)];
                size_t b = CIDX(D, i, j, 0);
                for (int k = 0; k < D->nl[2]; k++) {
                    u[b + k] += alpha * pp[k];
                    double rv = r[b + k] - alpha * q[b + k];
                    r[b + k] = rv;
                    rr += rv * rv;
                }
            }
        TICK(vec);
        if (pc != PC_NONE) { precond(D, pc, r, z); TICK(prec); }
        double rzl = 0.0;
        if (pc == PC_NONE) rzl = rr;
        else for (size_t i = 0; i < N; i++) rzl += r[i] * z[i];
        loc[0] = rzl; loc[1] = rr;
        TICK(vec);
        gsum_double(loc, red, 2);                   TICK(reduce);
        double beta = red[0] / rz;
        rz = red[0];
        UPDATE_P(beta);                             TICK(vec);
        info.res = sqrt(red[1]);
        info.iters++;
        if (hist) hist[info.iters] = info.res;
    }
    info.converged = info.res <= rtol * info.res0;
    T.total = msg_wtime() - tstart;
    if (tm) *tm = T;
    free(r); free(q); free(zb); free(p);
    return info;
}
