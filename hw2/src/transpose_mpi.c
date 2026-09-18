#include <stdlib.h>
#include <string.h>
#include "transpose_mpi.h"
#include "transpose.h"
#include "msg.h"

/* Profiling accumulators (see transpose_mpi.h).  Plain per-rank wall
 * time; msg_wtime() is a handful of ns and every bracketed region is
 * at least a memcpy over a block, so the instrumentation does not
 * perturb what it measures. */
static transpose_mpi_prof prof;

void transpose_mpi_prof_reset(void)
{
    prof.total = prof.msg = prof.pack = prof.local = 0.0;
}

void transpose_mpi_prof_get(transpose_mpi_prof *out)
{
    *out = prof;
}

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
    {
        const double t0 = msg_wtime();
        transpose_real(Mp,Qp, A_local + (size_t)i0y*Mp, B_local + (size_t)i0x*Qp);
        prof.local += msg_wtime()-t0;
    }
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
    double t0;

    t0 = msg_wtime();
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
    prof.msg += msg_wtime()-t0;

    t0 = msg_wtime();
    for (s=0; s<P; ++s) {
        int Is_start,Ms;
        if (s==rank) continue;
        block_bounds(nx,P,s,&Is_start,&Ms);
        transpose_real(Ms,Qp, rbuf[s], B_local + (size_t)Is_start*Qp);
        free(rbuf[s]);
    }
    prof.local += msg_wtime()-t0;
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
        double t0 = msg_wtime();
        isend(q, A_local + (size_t)Jq_start*Mp,
              (int)((size_t)Mp*Qq*sizeof(double)), 0);
        irecv(q, rbuf, (int)((size_t)Mq*Qp*sizeof(double)), 0);
        msgwait();
        prof.msg += msg_wtime()-t0;

        t0 = msg_wtime();
        transpose_real(Mq,Qp, rbuf, B_local + (size_t)Iq_start*Qp);
        prof.local += msg_wtime()-t0;
        free(rbuf);
    }
}

/*======================================================================
 *  Crystal Router (lec06; Fox et al., "Solving Problems on Concurrent
 *  Processors", 1988).
 *
 *  Instead of the P-1 direct exchanges of the dealing algorithms, the
 *  ranks are recursively bisected: ceil(log2(P)) rounds, and in each
 *  round every rank makes ONE exchange with a single partner across
 *  the current cut, shipping everything it holds that belongs on the
 *  far side -- its own blocks and whatever it is relaying for others.
 *  Messages per rank drop from P-1 to ~log2(P); total bytes rise,
 *  since a block bound for a distant rank is forwarded up to log2(P)
 *  times.  Latency-bound regimes (many ranks, small blocks) favour
 *  this; bandwidth-bound regimes favour dealing.
 *
 *  Because blocks are relayed by ranks they are not addressed to,
 *  each travels as a self-describing packet: a (src,dest) header
 *  followed by an Ms x Qdest payload.  Both dimensions come back out
 *  of block_bounds() from the header alone, so nothing else rides
 *  along, and an intermediate rank never has to interpret a payload.
 *
 *  Non-power-of-two P: the range [lo,hi) splits at mid = lo+(hi-lo)/2
 *  and rank lo+a pairs with mid+a.  An odd-length range leaves the
 *  upper half one longer, so its last rank (hi-1) has no partner; it
 *  ships its lower-half-bound packets to rank lo and receives nothing
 *  that round.  Nothing is stranded: packets addressed to hi-1 cross
 *  the cut through the regular pairs and reach it in the later rounds
 *  that bisect [mid,hi).
 *====================================================================*/

typedef struct { int src, dest; } cr_hdr;

/* The header is rounded up to whole doubles so payloads stay 8-byte
 * aligned and a packet pool can be indexed entirely in doubles. */
#define CR_HDR_D   ((sizeof(cr_hdr)+sizeof(double)-1)/sizeof(double))
#define CR_TAG_SIZE 1
#define CR_TAG_DATA 2

/* Packets stored back to back; `n` and `cap` count doubles. */
typedef struct { double *buf; size_t n, cap; } cr_pool;

static void cr_pool_init(cr_pool *p) { p->buf=0; p->n=0; p->cap=0; }
static void cr_pool_free(cr_pool *p) { free(p->buf); p->buf=0; p->n=0; p->cap=0; }

static void cr_pool_reserve(cr_pool *p, size_t need)
{
    size_t cap;
    if (need <= p->cap) return;
    cap = p->cap ? p->cap : 256;
    while (cap < need) cap *= 2;
    p->buf = (double*)realloc(p->buf, cap*sizeof(double));
    if (!p->buf) abort();
    p->cap = cap;
}

static void cr_pool_append(cr_pool *p, const double *pkt, size_t len)
{
    cr_pool_reserve(p, p->n + len);
    memcpy(p->buf + p->n, pkt, len*sizeof(double));
    p->n += len;
}

static void cr_pool_push(cr_pool *p, int src, int dest,
                          const double *payload, size_t np)
{
    cr_hdr h;
    h.src = src; h.dest = dest;
    cr_pool_reserve(p, p->n + CR_HDR_D + np);
    memcpy(p->buf + p->n, &h, sizeof h);
    memcpy(p->buf + p->n + CR_HDR_D, payload, np*sizeof(double));
    p->n += CR_HDR_D + np;
}

/* Total length of the packet src->dest: the block is Ms x Qdest, with
 * Ms rows from the SOURCE's x-range and Qdest columns from the
 * DESTINATION's y-range.  Any rank holding the packet can compute it. */
static size_t cr_packet_doubles(int nx, int ny, int P, int src, int dest)
{
    int s0,Ms,d0,Qd;
    block_bounds(nx,P,src, &s0,&Ms);
    block_bounds(ny,P,dest,&d0,&Qd);
    return CR_HDR_D + (size_t)Ms*Qd;
}

static void transpose_mpi_crystal(int nx, int ny, int P, int rank,
                                   const double * restrict A_local,
                                   double * restrict B_local)
{
    int i0x,Mp,i0y,Qp,q,lo=0,hi=P;
    size_t o;
    double tp;
    cr_pool pool;
    block_bounds(nx,P,rank,&i0x,&Mp);
    block_bounds(ny,P,rank,&i0y,&Qp);

    /* One packet per off-diagonal destination.  The part of A_local
     * bound for q is q's whole y-range over all Mp local x-rows, i.e.
     * Qq consecutive columns -- contiguous, so a straight copy. */
    cr_pool_init(&pool);
    tp = msg_wtime();
    for (q=0; q<P; ++q) {
        int Jq_start,Qq;
        if (q==rank) continue;
        block_bounds(ny,P,q,&Jq_start,&Qq);
        cr_pool_push(&pool,rank,q,A_local+(size_t)Jq_start*Mp,(size_t)Mp*Qq);
    }
    prof.pack += msg_wtime()-tp;

    while (hi-lo > 1) {
        const int nl    = (hi-lo)/2;
        const int mid   = lo+nl;
        const int lower = (rank < mid);
        int partner = -1;     /* two-way exchange partner, -1 = none   */
        int extra_src = -1;   /* receive-only partner (the odd rank)   */
        int extra_dst = -1;   /* send-only partner (I am the odd rank) */
        size_t nsend, nrecv_p = 0, nrecv_e = 0;
        cr_pool keep, out;

        if (lower) {
            partner = mid + (rank-lo);
            if (((hi-lo)&1) && rank==lo) extra_src = hi-1;
        } else if (rank-mid < nl) {
            partner = lo + (rank-mid);
        } else {
            extra_dst = lo;
        }

        /* Split the pool: everything addressed to the far side of the
         * cut goes out, the rest stays -- including packets addressed
         * to this rank, which are on this side by definition. */
        cr_pool_init(&keep);
        cr_pool_init(&out);
        tp = msg_wtime();
        for (o=0; o<pool.n; ) {
            cr_hdr h;
            size_t len;
            int far;
            memcpy(&h, pool.buf+o, sizeof h);
            len = cr_packet_doubles(nx,ny,P,h.src,h.dest);
            far = lower ? (h.dest >= mid) : (h.dest < mid);
            cr_pool_append(far ? &out : &keep, pool.buf+o, len);
            o += len;
        }
        prof.pack += msg_wtime()-tp;
        cr_pool_free(&pool);
        nsend = out.n;

        /* irecv needs the byte count up front and a relayed pool has
         * no size the receiver can predict, so every round is a size
         * exchange followed by the payload exchange.  Both sides then
         * agree on which transfers are empty and skip them together. */
        tp = msg_wtime();
        if (partner>=0) {
            isend(partner,&nsend,  (int)sizeof nsend,  CR_TAG_SIZE);
            irecv(partner,&nrecv_p,(int)sizeof nrecv_p,CR_TAG_SIZE);
        }
        if (extra_dst>=0) isend(extra_dst,&nsend,  (int)sizeof nsend,  CR_TAG_SIZE);
        if (extra_src>=0) irecv(extra_src,&nrecv_e,(int)sizeof nrecv_e,CR_TAG_SIZE);
        msgwait();

        /* Both incoming pools land directly at the end of `keep`;
         * reserve once up front so nothing reallocs under an irecv. */
        cr_pool_reserve(&keep, keep.n + nrecv_p + nrecv_e);
        if (partner>=0) {
            if (nsend)
                isend(partner,out.buf,(int)(nsend*sizeof(double)),CR_TAG_DATA);
            if (nrecv_p)
                irecv(partner,keep.buf+keep.n,
                      (int)(nrecv_p*sizeof(double)),CR_TAG_DATA);
        }
        if (extra_dst>=0 && nsend)
            isend(extra_dst,out.buf,(int)(nsend*sizeof(double)),CR_TAG_DATA);
        if (extra_src>=0 && nrecv_e)
            irecv(extra_src,keep.buf+keep.n+nrecv_p,
                  (int)(nrecv_e*sizeof(double)),CR_TAG_DATA);
        msgwait();
        prof.msg += msg_wtime()-tp;

        keep.n += nrecv_p + nrecv_e;
        cr_pool_free(&out);
        pool = keep;

        if (lower) hi = mid; else lo = mid;
    }

    /* The bisection has narrowed to this rank alone, so every packet
     * still held is addressed here: scatter each Ms x Qp block into
     * B_local's x-columns Is. */
    tp = msg_wtime();
    for (o=0; o<pool.n; ) {
        cr_hdr h;
        int Is_start,Ms;
        memcpy(&h, pool.buf+o, sizeof h);
        block_bounds(nx,P,h.src,&Is_start,&Ms);
        transpose_real(Ms,Qp, pool.buf+o+CR_HDR_D, B_local+(size_t)Is_start*Qp);
        o += CR_HDR_D + (size_t)Ms*Qp;
    }
    prof.local += msg_wtime()-tp;
    cr_pool_free(&pool);
}

void transpose_mpi(int nx, int ny,
                    const double * restrict A_local,
                    double * restrict B_local,
                    transpose_mpi_strategy strategy)
{
    const int P    = num_ranks();
    const int rank = msg_rank();
    const double t0 = msg_wtime();

    transpose_self(nx,ny,P,rank,A_local,B_local);
    if (P==1) { prof.total += msg_wtime()-t0; return; }

    switch (strategy) {
    case TRANSPOSE_MPI_DEALING_SIMULTANEOUS:
        transpose_mpi_dealing_simultaneous(nx,ny,P,rank,A_local,B_local);
        break;
    case TRANSPOSE_MPI_DEALING_STAGGERED:
        transpose_mpi_dealing_staggered(nx,ny,P,rank,A_local,B_local);
        break;
    case TRANSPOSE_MPI_CRYSTAL_ROUTER:
        transpose_mpi_crystal(nx,ny,P,rank,A_local,B_local);
        break;
    }

    prof.total += msg_wtime()-t0;
}
