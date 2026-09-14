#!/bin/bash
#SBATCH --account=26fa-cs598pa-eng
#SBATCH --partition=eng-instruction    # <- replace with your partition
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=poisson
#SBATCH --output=logfile

module load openmpi

mpirun -np 1 --map-by ppr:1:node --bind-to core ./test_poisson
