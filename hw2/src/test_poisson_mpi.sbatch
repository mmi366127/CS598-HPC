#!/bin/bash
#SBATCH --account=26fa-cs598pa-eng
#SBATCH --partition=eng-instruction
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --job-name=poisson_mpi
#SBATCH --output=logfile_poisson_mpi

module load openmpi

echo "== serial baseline (test_poisson, P=1) =="
mpirun -np 1 --map-by ppr:1:node --bind-to core ./test_poisson 128 512

for P in 1 2 4 8 16; do
    for strat in 1 0 2; do
        name=$(case $strat in 1) echo staggered;; 0) echo simultaneous;; 2) echo crystal;; esac)
        echo
        echo "== test_poisson_mpi  P=$P  strategy=$name =="
        mpirun -np $P --map-by ppr:$P:node --bind-to core \
            ./test_poisson_mpi 128 512 2 1 $strat
    done
done
