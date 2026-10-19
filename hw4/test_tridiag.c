#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "msg.h"
#include "tridiag_mpi.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double exact_u(double x, double Lx)
{
    return sin(M_PI*x/Lx);
}
static double rhs_f(double x, double Lx)
{
    return M_PI*M_PI/Lx/Lx*exact_u(x,Lx);
}

/* usage: test_tridiag [n] [reps]   (n interior grid points, default 128) */
int main(int argc, char **argv)
{
    int n = 128, reps = 1;
    double Lx = 1.0;

    msg_init(&argc, &argv);
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) reps = atoi(argv[2]);
    int r = msg_rank(), P = num_ranks();

    int start, k;
    tridiag_layout(n, P, r, &start, &k);
    if (k < 2 && r < P - 1) {
        if (r == 0) fprintf(stderr, "need n >= 2P\n");
        msg_finalize();
        return 1;
    }

    /* Dirichlet 1D Poisson: x_i = (i+1) h, h = Lx/(n+1), A = tridiag(-1,2,-1)/h^2 */
    double h = Lx / (n + 1), h2 = 1.0 / (h * h);
    double *buf = malloc(6 * (size_t)k * sizeof(double));
    double *a = buf, *b = a + k, *c = b + k, *f = c + k, *x = f + k;
    double *ue = x + k;
    for (int i = 0; i < k; i++) {
        double xi = (start + i + 1) * h;
        a[i] = 2 * h2; b[i] = -h2; c[i] = -h2;
        f[i] = rhs_f(xi, Lx);
        ue[i] = exact_u(xi, Lx);
    }

    tridiag_times t = {0}, tbest = {0};
    double best = 1e300;
    for (int it = 0; it < reps; it++) {
        msg_barrier();
        double t0 = msg_wtime();
        tridiag_solve_mpi(k, a, b, c, f, x, &t);
        double dt = msg_wtime() - t0;
        if (dt < best) { best = dt; tbest = t; }
    }

    /* residual ||Ax - f||_inf: halo exchange of neighbour boundary values */
    double lh = 0.0, rh = 0.0, xfirst = x[0], xlast = x[k-1];
    if (r > 0)     irecv(r - 1, &lh, sizeof lh, 1);
    if (r < P - 1) irecv(r + 1, &rh, sizeof rh, 2);
    if (r > 0)     isend(r - 1, &xfirst, sizeof xfirst, 2);
    if (r < P - 1) isend(r + 1, &xlast, sizeof xlast, 1);
    msgwait();

    
    double res = 0, fn = 0, err = 0, un = 0;
    for (int i = 0; i < k; i++) {
        double xm = i > 0 ? x[i-1] : lh, xp = i < k - 1 ? x[i+1] : rh;
        double ax = a[i] * x[i];
        if (start + i > 0)     ax += c[i] * xm;
        if (start + i < n - 1) ax += b[i] * xp;
        res = fmax(res, fabs(ax - f[i]));
        fn  = fmax(fn, fabs(f[i]));
        err = fmax(err, fabs(x[i] - ue[i]));
        un  = fmax(un, fabs(ue[i]));
    }

    /* max over ranks via zero-padded sum (rank slots) */
    double *s = calloc(14 * (size_t)P, sizeof(double)), *S = s + 7 * P;
    double v[7] = {res, fn, err, un, best, tbest.t_local, tbest.t_comm};
    for (int j = 0; j < 7; j++) s[j * P + r] = v[j];
    gsum_double(s, S, 7 * P);
    if (r == 0) {
        double mx[7] = {0};
        for (int j = 0; j < 7; j++)
            for (int q = 0; q < P; q++) mx[j] = fmax(mx[j], S[j * P + q]);
        printf("n=%d P=%d  ||Ax-f||/||f|| = %.3e  max|u-u_exact| = %.3e"
               "  (h^2 pi^2/12 ~ %.3e)\n", n, P, mx[0] / mx[1], mx[2] / mx[3],
               h * h * M_PI * M_PI / 12);
        printf("time %.6e s  [local %.3e  allgather %.3e  (rank0: reduced %.3e  back %.3e)]\n",
               mx[4], mx[5], mx[6], tbest.t_reduced, tbest.t_back);
    }
    free(s);
    free(buf);
    msg_finalize();
    return 0;
}
