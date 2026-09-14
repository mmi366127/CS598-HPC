MPICC ?= mpicc
CFLAGS_COMMON = -std=c99 -D_POSIX_C_SOURCE=199309L -Wall -Wextra
CFLAGS_OPT    = -O3 -march=native -ffast-math -fno-unroll-loops
LIBS = -lm

MAKEFLAGS += -r   # disable make's built-in implicit rules, so a config
                  # problem fails loudly instead of silently compiling
                  # with the wrong tool (see note in README).

# poisson.c now calls msg_wtime()/msg_rank() (see poisson.c), so
# msg_mpi.o goes into COMMON_OBJS -- every executable below links it,
# since all three share poisson.o through COMMON_OBJS.
COMMON_OBJS = block_fft.o block_fst.o transpose.o transpose_mpi.o poisson.o msg_mpi.o
EXES        = test_block_fst test_poisson bench_transpose test_transpose_mpi test_poisson_mpi verify_poisson

all: $(EXES) asm

test_block_fst: $(COMMON_OBJS) test_block_fst.o
	$(MPICC) $(CFLAGS_COMMON) $(CFLAGS_OPT) -o $@ $^ $(LIBS)

test_poisson: $(COMMON_OBJS) test_poisson.o
	$(MPICC) $(CFLAGS_COMMON) $(CFLAGS_OPT) -o $@ $^ $(LIBS)

bench_transpose: $(COMMON_OBJS) bench_transpose.o
	$(MPICC) $(CFLAGS_COMMON) $(CFLAGS_OPT) -o $@ $^ $(LIBS)

test_transpose_mpi: $(COMMON_OBJS) test_transpose_mpi.o
	$(MPICC) $(CFLAGS_COMMON) $(CFLAGS_OPT) -o $@ $^ $(LIBS)

# poisson_mpi.o is its own object (not in COMMON_OBJS) so the serial
# build path above is untouched -- poisson.c/poisson.o stay the
# unmodified serial reference used for correctness and as the P=1
# baseline for parallel-efficiency comparisons later.
test_poisson_mpi: $(COMMON_OBJS) poisson_mpi.o test_poisson_mpi.o
	$(MPICC) $(CFLAGS_COMMON) $(CFLAGS_OPT) -o $@ $^ $(LIBS)

verify_poisson: $(COMMON_OBJS) verify_poisson.o
	$(MPICC) $(CFLAGS_COMMON) $(CFLAGS_OPT) -o $@ $^ $(LIBS)

%.o: %.c
	$(MPICC) $(CFLAGS_COMMON) $(CFLAGS_OPT) -c $< -o $@

%.s: %.c
	$(MPICC) $(CFLAGS_COMMON) $(CFLAGS_OPT) -S $< -o $@

asm: block_fft.s block_fst.s transpose.s poisson.s

clean:
	rm -f *.o *.s $(EXES)

.PHONY: all asm clean
