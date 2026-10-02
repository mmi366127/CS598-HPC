CC = mpicc
CFLAGS = -O3 -march=native -Wall
test_lu: test_lu.c lu_mpi.c msg_mpi.c lu_mpi.h msg.h
	$(CC) $(CFLAGS) -o $@ test_lu.c lu_mpi.c msg_mpi.c -lm -lopenblas -lopenblas
clean:
	rm -f test_lu
