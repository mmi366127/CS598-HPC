# module load openmpi/5.0.1-gcc-13.3.0
MPICC  ?= mpicc
CFLAGS  = -std=gnu99 -O3 -march=native -Wall -Wextra

all: pcg_serial

pcg_serial: pcg_serial.c pcg.c pcg.h msg_mpi.c msg.h
	$(MPICC) $(CFLAGS) -o $@ pcg_serial.c pcg.c msg_mpi.c -lm

clean:
	rm -f pcg_serial

.PHONY: all clean
