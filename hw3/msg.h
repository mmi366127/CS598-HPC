#ifndef MSG_H
#define MSG_H

/*
 * RISC-style message-passing interface.
 *
 * Application code should include this file only.
 * MPI is hidden entirely in msg_mpi.c.
 *
 * Minimal instruction set:
 *
 *   msg_init / msg_finalize
 *   msg_rank / num_ranks
 *   msg_barrier / msg_wtime
 *   isend / irecv / msgwait
 *   gsum_double / bcast
 *
 * No request handles are exposed to application code.
 */

int    msg_init(int *argc, char ***argv);
void   msg_finalize(void);

int    msg_rank (void);
int    num_ranks(void);

double msg_wtime(void);
double msg_wtick(void);
void   msg_barrier(void);

/* Queue one nonblocking send/recv internally. */
void isend(int dest, const void *buf, int nbytes, int tag);
void irecv(int src,  void *buf,       int nbytes, int tag);

/* Wait for all internally queued nonblocking operations. */
void msgwait(void);

void gsum_double(const double *x, double *y, int n);
void bcast(void *buf, int nbytes, int root);

#endif
