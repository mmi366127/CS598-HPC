#include <stdlib.h>
#include "transpose_mpi.h"
#include "transpose.h"
#include "msg.h"

void block_bounds(int n, int P, int p, int *start, int *count)
{
    const int base = n / P;
    const int rem  = n % P;
    *count = base + (p < rem ? 1 : 0);
    *start = p*base + (p < rem ? p : rem);
}

/* Own diagonal block: A_local's own Jp (=own y-range) columns feed
 * B_local's own Ip (=own x-range) columns.  Purely local, no message. */
static void transpose_self(int nx, int ny, int P, int rank,
                            const double * restrict A_local,
                            double * restrict B_local)
{
    int i0x,Mp,i0y,Qp;
    (void)ny;
    block_bounds(nx,P,rank,&i0x,&Mp);
    block_bounds(ny,P,rank,&i0y,&Qp);
    transpose_real(Mp,Qp, A_local + (size_t)i0y*Mp, B_local + (size_t)i0x*Qp);
}

/* Dealing, simultaneous: post every irecv and every isend at once,
 * one msgwait.  lec05 p.6's "P-1 ranks sending to rank 0 at the same
 * time" hotspot -- milder here than with blocking sends, since
 * nothing actually blocks on it, but still full simultaneous load. */
static void transpose_mpi_dealing_simultaneous(int nx, int ny, int P, int rank,
                                 const double * restrict A_local,
                                 double * restrict B_local)
{
    int i0x,Mp,i0y,Qp,q,s;
    block_bounds(nx,P,rank,&i0x,&Mp);
    block_bounds(ny,P,rank,&i0y,&Qp);

    double **rbuf = (double**)calloc((size_t)P,sizeof(double*));

    for (q=0; q<P; ++q) {
        int Jq_start,Qq;
        if (q==rank) continue;
        block_bounds(ny,P,q,&Jq_start,&Qq);
        isend(q, A_local + (size_t)Jq_start*Mp,
              (int)((size_t)Mp*Qq*sizeof(double)), 0);
    }
    for (s=0; s<P; ++s) {
        int Is_start,Ms;
        if (s==rank) continue;
        block_bounds(nx,P,s,&Is_start,&Ms);
        rbuf[s] = (double*)malloc((size_t)Ms*Qp*sizeof(double));
        irecv(s, rbuf[s], (int)((size_t)Ms*Qp*sizeof(double)), 0);
    }
    msgwait();

    for (s=0; s<P; ++s) {
        int Is_start,Ms;
        if (s==rank) continue;
        block_bounds(nx,P,s,&Is_start,&Ms);
        transpose_real(Ms,Qp, rbuf[s], B_local + (size_t)Is_start*Qp);
        free(rbuf[s]);
    }
    free(rbuf);
}

/* Dealing, staggered: lec05 p.6's fix, realized with a real msgwait
 * per round (see the strategy-enum comment in transpose_mpi.h for why
 * that's needed with a non-blocking-only isend/irecv/msgwait API). */
static void transpose_mpi_dealing_staggered(int nx, int ny, int P, int rank,
                                    const double * restrict A_local,
                                    double * restrict B_local)
{
    int i0x,Mp,i0y,Qp,r;
    block_bounds(nx,P,rank,&i0x,&Mp);
    block_bounds(ny,P,rank,&i0y,&Qp);

    /* Round-robin "circle method": at round r, rank p's partner is
     * q = (r - p) mod P.  This is self-inverse -- partner(q,r) = p --
     * so both sides always agree on who they're exchanging with this
     * round, unlike partner = (p+r) mod P, which pairs up p->q but
     * leaves q looking for a message from someone else that round and
     * deadlocks for P not a power of two.  r runs 0..P-1 (P rounds,
     * not P-1); each rank sits out exactly one round, when q==rank. */
    for (r=0; r<P; ++r) {
        const int q = ((r-rank) % P + P) % P;
        int Jq_start,Qq,Iq_start,Mq;
        if (q==rank) continue;
        block_bounds(ny,P,q,&Jq_start,&Qq);
        block_bounds(nx,P,q,&Iq_start,&Mq);

        double *rbuf = (double*)malloc((size_t)Mq*Qp*sizeof(double));
        isend(q, A_local + (size_t)Jq_start*Mp,
              (int)((size_t)Mp*Qq*sizeof(double)), 0);
        irecv(q, rbuf, (int)((size_t)Mq*Qp*sizeof(double)), 0);
        msgwait();

        transpose_real(Mq,Qp, rbuf, B_local + (size_t)Iq_start*Qp);
        free(rbuf);
    }
}

void transpose_mpi(int nx, int ny,
                    const double * restrict A_local,
                    double * restrict B_local,
                    transpose_mpi_strategy strategy)
{
    const int P    = num_ranks();
    const int rank = msg_rank();

    transpose_self(nx,ny,P,rank,A_local,B_local);
    if (P==1) return;

    switch (strategy) {
    case TRANSPOSE_MPI_DEALING_SIMULTANEOUS:
        transpose_mpi_dealing_simultaneous(nx,ny,P,rank,A_local,B_local);
        break;
    case TRANSPOSE_MPI_DEALING_STAGGERED:
        transpose_mpi_dealing_staggered(nx,ny,P,rank,A_local,B_local);
        break;
    }
}
