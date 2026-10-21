#ifndef TRIDIAG_MPI_H
#define TRIDIAG_MPI_H

typedef struct {
    double t_local;
    double t_comm;
    double t_reduced;
    double t_back;
} tridiag_times; // For timings

/* Factor the local rows once; must precede tridiag_substitute_mpi. */
void tridiag_factor_mpi(int k, const double *a, const double *b,
                        const double *c, tridiag_times *t);

/* Solve one RHS using the last factorization (same k, b, c). */
void tridiag_substitute_mpi(int k, const double *b, const double *c,
                            const double *f, double *x, tridiag_times *t);

#endif
