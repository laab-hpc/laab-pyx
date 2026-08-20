#!/bin/bash
#SBATCH --job-name=CPU-14x
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1           
#SBATCH --cpus-per-task=14
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH -A hpc2n2026-184
#SBATCH -C skylake

set -euo pipefail

module purge
module load GCC/12.3.0 OpenMPI/4.1.5
module load PyTorch/2.1.2-CUDA-12.1.1

export LD_BLAS='-lopenblas'
export OMP_NUM_THREADS=14
export LAAB_REPS=10
export LAAB_N=3000

export LAAB_BUILD_DIR=$(pwd)/build

lscpu > cpu_info.txt

srun make --no-print-directory -C ../../src  | tee -a data.txt


