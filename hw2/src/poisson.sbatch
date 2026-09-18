#!/bin/bash
#SBATCH --account=26fa-cs598pa-eng
#SBATCH --partition=eng-instruction
#SBATCH --constraint=AE7713
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --job-name=poisson
#SBATCH --output=logfile

module load openmpi

# The Makefile compiles with -march=native, which is resolved against
# whatever CPU runs the compiler -- and the login node (Zen4, AVX-512)
# is NOT the architecture of every compute node here: the AE7713 nodes
# are Zen3 with no AVX-512, so a login-node build dies with SIGILL on
# them.  Build on the node that will run the job instead.  The
# -C AE7713 above keeps the allocation to one CPU model so that stays
# true for every node in it.
make -s clean && make -s || exit 1

# Nx = Ny are interval counts; nx = Nx-1 interior unknowns must satisfy
# 2*(nx+1) = 4^k.  Since Nx = nx+1 that means 2*Nx = 4^j, i.e. Nx is a
# power of two with an ODD exponent:
#     Nx in {32, 128, 512, 2048, 8192, 32768, 131072, ...}  =  2^(2j-1)
# The even exponents (64, 256, 1024, 4096, ...) are NOT representable:
# 2*64 = 128 and 2*256 = 512 are powers of two but not powers of four,
# and block_fft is radix-4 only.
SIZES=${SIZES:-"32 128 512 2048 8192"}
RANKS=${RANKS:-"1 2 4 8 16 32 64 128 256"}
STRATS=${STRATS:-"1 2"}       # 1 = dealing-staggered, 2 = crystal-router

# SPLIT_NODES=n spreads EVERY run over exactly n nodes with P/n ranks on
# each, so rank count and node count vary independently and the
# interconnect is in the path at every P.  Left at 1 (the default) the
# mapping is by slot, which fills one node before spilling onto the next
# -- what the original single-node sweep did.
#
# A P that does not divide evenly by SPLIT_NODES (P=1 in practice) falls
# back to by-slot and so lands on ONE node; the placement table says so.
SPLIT_NODES=${SPLIT_NODES:-1}

# Rank placement across nodes.  VERIFIED on this cluster (job 10608853):
#   --npernode N --bind-to none        spreads correctly, real MPI works
#   --npernode N --bind-to core|socket|hwthread|...   ALL fail, exit 213
#   srun --cpu-bind=cores              launches SINGLETONS, not an MPI job
#                                      ("No PMIx server was reachable")
# The cause shows up under --bind-to core:overload-allowed:
#   "we found no available cpus on Node: ccc0393" -- prterun cannot see
# the remote node's cpuset, so NO binding policy works across nodes.
#
# Consequence: a multi-node run must be unpinned.  To keep single-node
# and multi-node runs comparable, set BIND=none for BOTH rather than
# pinning one and not the other.
BIND=${BIND:-core}

launch() {            # launch <P> <argv...>
    local P=$1; shift
    if [ "$SPLIT_NODES" -gt 1 ] && [ $((P % SPLIT_NODES)) -eq 0 ]; then
        mpirun -np "$P" --npernode $((P / SPLIT_NODES)) --bind-to "$BIND" "$@"
    else
        mpirun -np "$P" --bind-to "$BIND" "$@"
    fi
}

# Self-check: run plain `hostname` under each mapping used below and
# tally ranks per node, so the log carries hard evidence of the
# placement rather than a claim about it.
echo "===== rank -> node placement (SPLIT_NODES=$SPLIT_NODES) ====="
for P in $RANKS; do
    if [ "$SPLIT_NODES" -gt 1 ] && [ $((P % SPLIT_NODES)) -eq 0 ]; then
        printf 'P=%-4s npernode %-4s bind=%-5s -> ' "$P" "$((P / SPLIT_NODES))" "$BIND"
    else
        printf 'P=%-4s by-slot       bind=%-5s -> ' "$P" "$BIND"
    fi
    # prterun on this cluster prints an ALLOCATED-NODES banner on stdout,
    # so keep only real `hostname` output: lines that ARE a node name.
    tally=$(launch "$P" hostname 2>/dev/null | grep -Eo '^ccc[0-9]+' | sort | uniq -c)
    echo "$tally" | awk '{printf "%s x%s  ", $2, $1}'
    got=$(echo "$tally" | grep -c .)
    tot=$(echo "$tally" | awk '{s+=$1} END{print s+0}')
    want=$SPLIT_NODES
    [ $((P % SPLIT_NODES)) -ne 0 ] && want=1
    if [ "$tot" != "$P" ] || [ "$got" != "$want" ]; then
        echo "  <-- WRONG PLACEMENT (wanted $P ranks on $want node(s)); aborting"
        exit 1
    fi
    echo
done
echo

for N in $SIZES; do
    # Reps are scaled down for the big grids: one 8192^2 solve is ~10 s
    # serial, and the run-to-run spread there is far below the cost of
    # more samples.
    case $N in
        131072|524288) NREP=1 ;;
        32768) NREP=2 ;;
        8192)  NREP=3 ;;
        2048)  NREP=5 ;;
        *)     NREP=10 ;;
    esac
    [ -n "$NREP_FORCE" ] && NREP=$NREP_FORCE
    for P in $RANKS; do
        # block_bounds must give every rank at least one x-row and one
        # y-row: P > nx would leave a rank empty, which
        # block_fst_plan_init rejects (it requires m >= 1).
        if [ "$P" -gt "$((N-1))" ]; then
            echo "BENCH $(printf '%6d %6d %4d' $N $N $P) -  SKIPPED (P > nx = $((N-1)))"
            continue
        fi
        for s in $STRATS; do
            launch "$P" ./bench_poisson_mpi "$N" "$N" "$s" "$NREP"
        done
    done
done

echo "SWEEP-COMPLETE"
