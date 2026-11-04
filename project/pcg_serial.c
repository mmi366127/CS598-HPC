/* Serial PCG baseline for the 3D Poisson problem on [0,L]^3, L = 1.
 *
 * case=sin : f = sin(pi x/L) sin(pi y/L) sin(pi z/L), u = 0 on all faces,
 *            vertex-centered grid, exact u = f / (3 pi^2 / L^2).
 * case=face: f = 0, u = 0 on x = 0, u = 1 on x = L, zero normal derivative on the
 *            four side faces (y = 0, L; z = 0, L).  Cell-centered grid, h = L/n.
 *            Exact u = x/L (the discrete scheme reproduces it exactly).
 *
 * case=cos : as face, but u(0,y,z) = 0 and u(L,y,z) = cos(pi y/L) cos(pi z/L).  This
 *            is an eigenmode of the side-face Neumann problem, so the exact solution is
 *            u = sinh(kx)/sinh(kL) cos(pi y/L) cos(pi z/L), k = sqrt(2) pi / L.
 * case=sincos: as face, but u(0,y,z) = sin(pi y/L) sin(pi z/L), u(L,y,z) = cos(pi y/L)
 *            cos(pi z/L).  No closed form (the sine data does not satisfy the side
 *            Neumann condition); check by mesh refinement.
 *
 * usage: pcg_serial [n=64] [pc=none|jacobi|ssor] [rtol=1e-10] [maxit=10000] [case=sin|face|cos|sincos]
 * Writes convergence.csv (iter, residual) and, for the non-sin cases, slice.csv
 * (u on the plane z = L/4, rows = x, columns = y). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "pcg.h"
#include "msg.h"

int main(int argc, char **argv)
{
    msg_init(&argc, &argv);
    int n = argc > 1 ? atoi(argv[1]) : 64;
    const char *pcs = argc > 2 ? argv[2] : "ssor";
    double rtol = argc > 3 ? atof(argv[3]) : 1e-10;
    int maxit = argc > 4 ? atoi(argv[4]) : 10000;
    const char *cname = argc > 5 ? argv[5] : "sin";
    int cs = !strcmp(cname, "face") ? 1 : !strcmp(cname, "cos") ? 2 : !strcmp(cname, "sincos") ? 3 : 0;
    int face = cs != 0;
    pc_t pc = !strcmp(pcs, "none") ? PC_NONE : !strcmp(pcs, "jacobi") ? PC_JACOBI : PC_SSOR;

    const double L = 1.0, PI = 3.14159265358979323846;
    pcg_prob P = { n, 0.0, {0, 0, 0} };
    if (face) { P.h = L / n; P.g[0] = -1; P.g[1] = P.g[2] = 1; }
    else        P.h = L / (n + 1);
    double h = P.h, x0 = face ? 0.5 * h : h;      /* coordinate of cell 0 */
    size_t N = (size_t)n * n * n;
    double *f = malloc(N * sizeof(double)), *u = calloc(N, sizeof(double));
    double *hist = malloc((maxit + 1) * sizeof(double));
    #define XC(i) (x0 + (i) * h)
    #define BC0(y, z) (cs == 3 ? sin(PI*(y)/L) * sin(PI*(z)/L) : 0.0)            /* u at x = 0 */
    #define BCL(y, z) (cs == 1 ? 1.0 : cos(PI*(y)/L) * cos(PI*(z)/L))            /* u at x = L */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                f[((size_t)i * n + j) * n + k] = face ? 2.0 / (h * h) * ((i == n - 1 ? BCL(XC(j), XC(k)) : 0.0) +
                                          (i == 0 ? BC0(XC(j), XC(k)) : 0.0)) :
                    sin(PI * (x0 + i * h) / L) * sin(PI * (x0 + j * h) / L) *
                    sin(PI * (x0 + k * h) / L);

    double t0 = msg_wtime();
    pcg_info info = pcg_solve(&P, f, u, pc, rtol, maxit, hist);
    double t = msg_wtime() - t0;

    double lam = 3.0 * PI * PI / (L * L), err = 0.0;   /* max-norm error */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++) {
                size_t p = ((size_t)i * n + j) * n + k;
                double kk = sqrt(2.0) * PI / L;
                double ex = cs == 0 ? f[p] / lam
                          : cs == 1 ? XC(i) / L
                          : cs == 2 ? sinh(kk * XC(i)) / sinh(kk * L) * cos(PI * XC(j) / L) * cos(PI * XC(k) / L)
                          : NAN;
                if (cs == 3) continue;
                err = fmax(err, fabs(u[p] - ex));
            }

    int mid = n / 2;
    printf("n=%d case=%s pc=%s iters=%d converged=%d |r|/|r0|=%.3e max_err=%.3e time=%.3fs u[n/2,n/2,n/2]=%.10f\n",
           n, cname, pcs, info.iters, info.converged, info.res / info.res0, err, t,
           u[((size_t)mid * n + mid) * n + mid]);
    if (face) {
        int m = n / 4;
        FILE *fs = fopen("slice.csv", "w");     /* u(x, y, z = cell m), rows = x */
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                fprintf(fs, "%.10e%c", u[((size_t)i * n + j) * n + m], j < n - 1 ? ',' : '\n');
        fclose(fs);
    }
    FILE *fp = fopen("convergence.csv", "w");
    fprintf(fp, "iter,residual\n");
    for (int k = 0; k <= info.iters; k++) fprintf(fp, "%d,%.16e\n", k, hist[k]);
    fclose(fp);
    free(f); free(u); free(hist);
    msg_finalize();
    return 0;
}
