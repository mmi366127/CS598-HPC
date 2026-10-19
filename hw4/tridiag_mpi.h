#ifndef TRIDIAG_MPI_H
#define TRIDIAG_MPI_H

typedef struct {
    double t_local;
    double t_comm;
    double t_reduced;
    double t_back;
} tridiag_times;

void tridiag_solve_mpi(int k, const double *a, const double *b,
                       const double *c, const double *f, double *x,
                       tridiag_times *t);

void tridiag_thomas(int n, const double *a, const double *b, const double *c,
                    const double *f, double *x, double *work);

#endif
