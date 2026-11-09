# module load openmpi/5.0.1-gcc-13.3.0
MPICC  ?= mpicc
CFLAGS  = -std=gnu99 -O3 -march=native -Wall -Wextra
MSG     = msg_mpi.c msg.h

all: pcg_serial pcg_mpi

pcg_serial: pcg_serial.c pcg.c pcg.h cases.h $(MSG)
	$(MPICC) $(CFLAGS) -o $@ pcg_serial.c pcg.c msg_mpi.c -lm

pcg_mpi: pcg_mpi_main.c pcg_mpi.c pcg_mpi.h pcg.h cases.h $(MSG)
	$(MPICC) $(CFLAGS) -o $@ pcg_mpi_main.c pcg_mpi.c msg_mpi.c -lm

clean:
	rm -f pcg_serial pcg_mpi

.PHONY: all clean
