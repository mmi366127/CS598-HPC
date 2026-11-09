/* Serial PCG baseline for the 3D Poisson problem (cases in cases.h).
 * usage: pcg_serial [n=64] [pc=none|jacobi|ssor] [rtol=1e-10] [maxit=10000] [case=sin|face|cos|sincos]
 * Writes convergence.csv (iter, residual) and, for the non-sin cases, slice.csv
 * (u on the plane z = L/4, rows = x, columns = y). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "pcg.h"
#include "cases.h"
#include "msg.h"

int main(int argc, char **argv)
{
    msg_init(&argc, &argv);
    int n = argc > 1 ? atoi(argv[1]) : 64;
    const char *pcs = argc > 2 ? argv[2] : "ssor";
    double rtol = argc > 3 ? atof(argv[3]) : 1e-10;
    int maxit = argc > 4 ? atoi(argv[4]) : 10000;
    const char *cname = argc > 5 ? argv[5] : "sin";
    int cs = case_parse(cname);
    pc_t pc = !strcmp(pcs, "none") ? PC_NONE : !strcmp(pcs, "jacobi") ? PC_JACOBI : PC_SSOR;

    pcg_prob P = case_problem(cs, n);
    size_t N = (size_t)n * n * n;
    double *f = malloc(N * sizeof(double)), *u = calloc(N, sizeof(double));
    double *hist = malloc((maxit + 1) * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                f[((size_t)i * n + j) * n + k] = case_rhs(cs, &P, i, j, k);

    double t0 = msg_wtime();
    pcg_info info = pcg_solve(&P, f, u, pc, rtol, maxit, hist);
    double t = msg_wtime() - t0;

    double err = 0.0, sum = 0.0, nrm = 0.0;
    int has_exact = !isnan(case_exact(cs, &P, 0, 0, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++) {
                double v = u[((size_t)i * n + j) * n + k];
                if (has_exact) err = fmax(err, fabs(v - case_exact(cs, &P, i, j, k)));
                sum += v; nrm += v * v;
            }

    printf("n=%d case=%s pc=%s iters=%d converged=%d relres=%.3e max_err=%.3e sum=%.10e norm2=%.10e time=%.3f\n",
           n, cname, pcs, info.iters, info.converged, info.res / info.res0,
           has_exact ? err : NAN, sum, sqrt(nrm), t);
    if (cs != CASE_SIN) {
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
