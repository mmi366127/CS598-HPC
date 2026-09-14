#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "msg.h"

#ifndef MSG_MAX_REQUESTS
#define MSG_MAX_REQUESTS 4096
#endif

static MPI_Comm msg_comm = MPI_COMM_WORLD;
static MPI_Request reqs[MSG_MAX_REQUESTS];
static int nreq = 0;

int msg_init(int *argc, char ***argv)
{
    int ierr = MPI_Init(argc,argv);
    msg_comm = MPI_COMM_WORLD;
    nreq = 0;
    return ierr;
}

void msg_finalize(void)
{
    MPI_Finalize();
}

int msg_rank(void)
{
    int r;
    MPI_Comm_rank(msg_comm,&r);
    return r;
}

int num_ranks(void)
{
    int s;
    MPI_Comm_size(msg_comm,&s);
    return s;
}

double msg_wtick(void)
{
    return MPI_Wtick();
}

double msg_wtime(void)
{
    return MPI_Wtime();
}

void msg_barrier(void)
{
    MPI_Barrier(msg_comm);
}

static void push_request(MPI_Request req)
{
    if (nreq >= MSG_MAX_REQUESTS) {
        fprintf(stderr,"msg: too many outstanding requests; increase MSG_MAX_REQUESTS\n");
        MPI_Abort(msg_comm,2);
    }

    reqs[nreq++] = req;
}

void isend(int dest, const void *buf, int nbytes, int tag)
{
    MPI_Request req;
    MPI_Isend(buf,nbytes,MPI_BYTE,dest,tag,msg_comm,&req);
    push_request(req);
}

void irecv(int src, void *buf, int nbytes, int tag)
{
    MPI_Request req;
    MPI_Irecv(buf,nbytes,MPI_BYTE,src,tag,msg_comm,&req);
    push_request(req);
}

void msgwait(void)
{
    if (nreq > 0) {
        MPI_Waitall(nreq,reqs,MPI_STATUSES_IGNORE);
        nreq = 0;
    }
}

void gsum_double(const double *x, double *y, int n)
{
    MPI_Allreduce(x,y,n,MPI_DOUBLE,MPI_SUM,msg_comm);
}

void bcast(void *buf, int nbytes, int root)
{
    MPI_Bcast(buf,nbytes,MPI_BYTE,root,msg_comm);
}
