/* Distributed PCG driver.
 * usage: pcg_mpi [n=64] [pc=none|jacobi|ssor] [rtol=1e-10] [maxit=10000]
 *                [case=sin|face|cos|sincos] [grid=auto|PXxPYxPZ] [reps=1]
 * Runs the solve `reps` times and reports the fastest.  rtol=0 runs exactly maxit
 * iterations (for timing per iteration).  Prints one line from rank 0; times are the
 * maximum over ranks.  Add "csv=FILE" as 8th argument to dump ||r_k||. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "pcg_mpi.h"
#include "cases.h"
#include "msg.h"

/* max over ranks of m values (v in/out), via sum-allreduce of a P x m table */
static void gmax(double *v, int m)
{
    int P = num_ranks(), p = msg_rank();
    double *a = calloc(2 * (size_t)P * m, sizeof(double)), *b = a + (size_t)P * m;
    for (int i = 0; i < m; i++) a[(size_t)i * P + p] = v[i];
    gsum_double(a, b, P * m);
    for (int i = 0; i < m; i++) {
        double mx = -INFINITY;
        for (int r = 0; r < P; r++) mx = fmax(mx, b[(size_t)i * P + r]);
        v[i] = mx;
    }
    free(a);
}

int main(int argc, char **argv)
{
    msg_init(&argc, &argv);
    int n = argc > 1 ? atoi(argv[1]) : 64;
    const char *pcs = argc > 2 ? argv[2] : "none";
    double rtol = argc > 3 ? atof(argv[3]) : 1e-10;
    int maxit = argc > 4 ? atoi(argv[4]) : 10000;
    const char *cname = argc > 5 ? argv[5] : "sin";
    const char *gs = argc > 6 ? argv[6] : "auto";
    int reps = argc > 7 ? atoi(argv[7]) : 1;
    const char *csv = argc > 8 ? argv[8] : NULL;
    int cs = case_parse(cname), rank = msg_rank(), P = num_ranks();
    pc_t pc = !strcmp(pcs, "none") ? PC_NONE : !strcmp(pcs, "jacobi") ? PC_JACOBI : PC_SSOR;

    int dims[3];
    if (!strcmp(gs, "auto")) pcg_choose_grid(n, P, dims);
    else if (sscanf(gs, "%dx%dx%d", &dims[0], &dims[1], &dims[2]) != 3 ||
             dims[0] * dims[1] * dims[2] != P) {
        if (rank == 0) fprintf(stderr, "bad grid '%s' for %d ranks\n", gs, P);
        msg_finalize(); return 1;
    }

    pcg_prob prob = case_problem(cs, n);
    pcg_dist D;
    pcg_dist_init(&D, &prob, dims);
    double *f = malloc(D.N * sizeof(double)), *u = malloc(D.N * sizeof(double));
    double *hist = malloc((maxit + 1) * sizeof(double));
    for (int i = 0; i < D.nl[0]; i++)
        for (int j = 0; j < D.nl[1]; j++)
            for (int k = 0; k < D.nl[2]; k++)
                f[((size_t)i * D.nl[1] + j) * D.nl[2] + k] =
                    case_rhs(cs, &prob, D.off[0] + i, D.off[1] + j, D.off[2] + k);

    pcg_info info = {0}, best_info = {0};
    double best[6] = {1e300}, tt[6];
    for (int rep = 0; rep < reps; rep++) {
        memset(u, 0, D.N * sizeof(double));
        pcg_times tm;
        msg_barrier();
        info = pcg_dist_solve(&D, f, u, pc, rtol, maxit, hist, &tm);
        tt[0] = tm.total; tt[1] = tm.halo; tt[2] = tm.apply; tt[3] = tm.vec; tt[4] = tm.prec; tt[5] = tm.reduce;
        gmax(tt, 6);
        if (tt[0] < best[0]) { memcpy(best, tt, sizeof best); best_info = info; }
    }
    info = best_info;

    /* solution statistics (u is from the last repetition; identical across reps) */
    double err = 0.0, st[2] = {0, 0}, sg[2];
    int has_exact = !isnan(case_exact(cs, &prob, 0, 0, 0));
    for (int i = 0; i < D.nl[0]; i++)
        for (int j = 0; j < D.nl[1]; j++)
            for (int k = 0; k < D.nl[2]; k++) {
                double v = u[((size_t)i * D.nl[1] + j) * D.nl[2] + k];
                if (has_exact)
                    err = fmax(err, fabs(v - case_exact(cs, &prob, D.off[0] + i, D.off[1] + j, D.off[2] + k)));
                st[0] += v; st[1] += v * v;
            }
    gsum_double(st, sg, 2);
    gmax(&err, 1);

    if (rank == 0) {
        printf("n=%d P=%d grid=%dx%dx%d case=%s pc=%s iters=%d converged=%d relres=%.3e max_err=%.3e "
               "sum=%.10e norm2=%.10e time=%.4e t_iter=%.4e t_halo=%.4e t_apply=%.4e t_vec=%.4e "
               "t_prec=%.4e t_reduce=%.4e\n",
               n, P, dims[0], dims[1], dims[2], cname, pcs, info.iters, info.converged,
               info.res / info.res0, has_exact ? err : NAN, sg[0], sqrt(sg[1]),
               best[0], best[0] / (info.iters ? info.iters : 1), best[1], best[2], best[3], best[4], best[5]);
        if (csv) {
            FILE *fp = fopen(csv, "w");
            fprintf(fp, "iter,residual\n");
            for (int k = 0; k <= info.iters; k++) fprintf(fp, "%d,%.16e\n", k, hist[k]);
            fclose(fp);
        }
    }
    pcg_dist_free(&D);
    free(f); free(u); free(hist);
    msg_finalize();
    return 0;
}
