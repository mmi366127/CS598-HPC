#ifndef TRIDIAG_MPI_H
#define TRIDIAG_MPI_H

/*
 * Distributed tridiagonal solver (no pivoting; A SPD or diagonally dominant).
 *
 * Global rows are cut into P contiguous chunks. Chunk p holds m_p interior
 * rows followed (for p < P-1) by the row of separator s_p, so
 * k_p = m_p + (p < P-1). Rank p stores the rows of its chunk:
 *   a[i] = A(i,i),  b[i] = A(i,i+1),  c[i] = A(i,i-1),  f[i] = rhs
 * (c of the first global row and b of the last global row are ignored).
 */

typedef struct {
    double t_local;    /* local factor + 3 solves + Schur contribution */
    double t_comm;     /* allgather of the 2x2 Schur pieces            */
    double t_reduced;  /* redundant (P-1) tridiagonal solve            */
    double t_back;     /* local back substitution                      */
} tridiag_times;

/* Chunk layout of n rows over P ranks: start index and size k of rank p. */
void tridiag_layout(int n, int P, int p, int *start, int *k);

/* Solve for this rank's chunk: x has k entries (interior then separator).
 * Requires k >= 2 on every rank but the last (m >= 1). t may be NULL. */
void tridiag_solve_mpi(int k, const double *a, const double *b,
                       const double *c, const double *f, double *x,
                       tridiag_times *t);

/* Serial Thomas algorithm: solves in place of x, work needs 1 double per row. */
void tridiag_thomas(int n, const double *a, const double *b, const double *c,
                    const double *f, double *x, double *work);

#endif
