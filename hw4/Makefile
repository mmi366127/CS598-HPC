# module load openmpi/5.0.1-gcc-13.3.0
MPICC  ?= mpicc
CFLAGS  = -std=gnu99 -O3 -march=native -Wall -Wextra

all: test_tridiag

test_tridiag: test_tridiag.c tridiag_mpi.c tridiag_mpi.h msg_mpi.c msg.h
	$(MPICC) $(CFLAGS) -o $@ test_tridiag.c tridiag_mpi.c msg_mpi.c -lm

clean:
	rm -f test_tridiag

.PHONY: all clean
