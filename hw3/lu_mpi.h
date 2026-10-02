#ifndef LU_MPI_H
#define LU_MPI_H

/*
 * Distributed LU (no pivoting) on a (P,Q) process grid, 2D block-cyclic
 * ("scattered") layout with nb x nb blocks. Rank r sits at grid position
 * (r / Q, r % Q). Global block (I,J) lives on process (I % P, J % Q).
 * Local storage is row-major, ld = nloc.
 */
typedef struct {
    int n, nb, P, Q, p, q;
    int mloc, nloc;     /* local rows / cols */
    double *a;          /* mloc x nloc, row-major */
} lu_dist;

/* number of local indices (of n, block nb, proc coord c of C) below global index g */
int lu_local_below(int g, int n, int nb, int c, int C);
/* global index of local index li */
int lu_global(int li, int nb, int c, int C);

void lu_dist_alloc(lu_dist *d, int n, int nb, int P, int Q, int rank);
void lu_dist_free(lu_dist *d);
void lu_dist_fill(lu_dist *d, double (*f)(int i, int j));

/* In-place LU: L (unit lower) and U stored in a. */
void lu_factor_mpi(lu_dist *d);

/* Gather full n x n matrix (row-major) on rank 0; returns NULL elsewhere. */
double *lu_gather(const lu_dist *d);

/* Serial reference, in-place, row-major n x n, no pivoting. */
void lu_factor_serial(double *a, int n);

#endif
