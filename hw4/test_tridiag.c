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

int main(int argc, char **argv)
{
    int n = 128, reps = 1;

    msg_init(&argc, &argv);
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) reps = atoi(argv[2]);
    int p = msg_rank(), P = num_ranks();

    if (n < 2 * P) {    /* every non-last rank needs >= 1 interior row + separator */
        if (p == 0) fprintf(stderr, "need n >= 2P (n=%d, P=%d)\n", n, P);
        msg_finalize();
        return 1;
    }

    double Lx = 1.0;
    double h = Lx / (n + 1), h2 = 1.0 / (h * h);
    int q = n / P, rem = n % P;
    int local_size = q + (p < rem);
    int start_idx = p * q + (p < rem ? p : rem);


    // Init a,b,c,f,u_exact
    double *a = malloc((size_t)local_size * sizeof(double));
    double *b = malloc((size_t)local_size * sizeof(double));
    double *c = malloc((size_t)local_size * sizeof(double));
    double *f = malloc((size_t)local_size * sizeof(double));
    double *x = malloc((size_t)local_size * sizeof(double));
    double *u_exact = malloc((size_t)local_size * sizeof(double));

    for (int i = 0; i < local_size; i++) {
        double xi = (start_idx + i + 1) * h;
        a[i] = 2 * h2; b[i] = -h2; c[i] = -h2;
        f[i] = rhs_f(xi, Lx);
        u_exact[i] = exact_u(xi, Lx);
    }


    // Solve, logging the best time
    tridiag_times t = {0}, tbest = {0};
    double best = 1e300;
    for (int it = 0; it < reps; it++) {
        msg_barrier();
        double t0 = msg_wtime();
        tridiag_factor_mpi(local_size, a, b, c, &t);
        tridiag_substitute_mpi(local_size, b, c, f, x, &t);
        double dt = msg_wtime() - t0;
        if (dt < best) { best = dt; tbest = t; }
    }


    // calculate error
    double lh = 0.0, rh = 0.0, ufirst = u[0], ulast = u[local_size-1];
    if (r > 0)     irecv(r - 1, &lh, sizeof lh, 1);
    if (r < P - 1) irecv(r + 1, &rh, sizeof rh, 2);
    if (r > 0)     isend(r - 1, &ufirst, sizeof ufirst, 2);
    if (r < P - 1) isend(r + 1, &ulast, sizeof ulast, 1);
    msgwait();

    double res = 0, fn = 0, err = 0, un = 0;
    for (int i = 0; i < local_size; i++) {
        double um = i > 0 ? u[i-1] : lh;
        double up = i < local_size - 1 ? u[i+1] : rh;
        double au = a[i] * u[i];
        if (start_idx + i > 0)     au += c[i] * um;
        if (start_idx + i < n - 1) au += b[i] * up;
        res = fmax(res, fabs(ax - f[i]));
        fn  = fmax(fn, fabs(f[i]));
        err = fmax(err, fabs(u[i] - u_exact[i]));
        un  = fmax(un, fabs(u_exact[i]));
    }

    /* max over ranks via zero-padded sum (rank slots) */
    double *s = calloc(14 * (size_t)P, sizeof(double)), *S = s + 7 * P;
    double v[7] = {res, fn, err, un, best, tbest.t_local, tbest.t_comm};
    for (int j = 0; j < 7; j++) s[j * P + p] = v[j];
    gsum_double(s, S, 7 * P);
    if (p == 0) {
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
    free(a);
    free(b);
    free(c);
    free(f);
    free(x);
    free(u_exact);

    msg_finalize();
    return 0;
}
