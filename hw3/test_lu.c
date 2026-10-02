/* usage: mpirun -np P*Q ./test_lu [N=129] [P=1] [Q=ranks/P] [check=1]
 * Test matrix: A = S + a I, S_ij = sqrt(2/N) sin(pi i j / N), a = N, n = N-1. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lu_mpi.h"
#include "msg.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int gN;
static double entry(int i, int j)   /* 0-based -> 1-based */
{
    double s = sqrt(2.0/gN) * sin(M_PI * (double)((long)(i+1)*(j+1) % (2L*gN)) / gN);
    return s + (i == j ? (double)gN : 0.0);
}

int main(int argc, char **argv)
{
    msg_init(&argc, &argv);
    const int rank = msg_rank(), np = num_ranks();
    int N  = argc > 1 ? atoi(argv[1]) : 129;
    int P  = argc > 2 ? atoi(argv[2]) : 2;
    int Q  = argc > 3 ? atoi(argv[3]) : np / P;
    int check = argc > 4 ? atoi(argv[4]) : 1;
    if (P*Q != np) {
        if (!rank) fprintf(stderr, "P*Q (%d*%d) != ranks (%d)\n", P, Q, np);
        msg_finalize(); return 1;
    }
    gN = N;
    int n = N - 1;

    lu_dist d;
    lu_dist_alloc(&d, n, P, Q, rank);
    lu_dist_fill(&d, entry);

    msg_barrier();
    double t = msg_wtime();
    lu_factor_mpi(&d);
    msg_barrier();
    t = msg_wtime() - t;

    double *G = lu_gather(&d);
    if (!rank) {
        printf("n=%d grid=%dx%d time=%.4fs GFLOPS=%.2f", n, P, Q, t,
               (2.0/3.0)*n*(double)n*n/t*1e-9);
        if (check) {
            /* ||A - LU||_max / ||A||_max, and diff vs serial LU */
            double *S = malloc(sizeof(double)*(size_t)n*n);
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) S[(size_t)i*n+j] = entry(i,j);
            double err = 0, amax = 0, diff = 0;
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
                double s = 0; int m = i < j ? i : j;
                for (int k = 0; k <= m; k++)
                    s += (k == i ? 1.0 : G[(size_t)i*n+k]) * G[(size_t)k*n+j];
                double e = fabs(s - S[(size_t)i*n+j]);
                if (e > err) err = e;
                if (fabs(S[(size_t)i*n+j]) > amax) amax = fabs(S[(size_t)i*n+j]);
            }
            lu_factor_serial(S, n);
            for (size_t i = 0; i < (size_t)n*n; i++) {
                double e = fabs(S[i]-G[i]); if (e > diff) diff = e;
            }
            printf(" resid=%.2e vs_serial=%.2e %s", err/amax, diff,
                   err/amax < 1e-12 ? "PASS" : "FAIL");
            free(S);
        }
        printf("\n");
        free(G);
    }
    lu_dist_free(&d);
    msg_finalize();
    return 0;
}
