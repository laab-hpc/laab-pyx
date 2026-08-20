#!/bin/bash

module purge
module load GCC/14.3.0 SciPy-bundle
source /proj/nobackup/aravind/LAAB-HPC/benchmarks/laab-pyx/venv/bin/activate

mkdir -p ../../../profiles

pyx-process-logs data.txt --outfile profile.pkl --name "CPU-4x" --system "kebnekaise:skylake" \
    --lib "PyTorch" --version "2.1.2" --toolchain "foss-2023a" \
    --config_path ../../src/config.json --prob_siz "3000" --reps "10" --omp_num_threads "4"

pyx-prepare-md-report profile.pkl --outfile README.md

mv profile.pkl ../../../profiles/profile_4_3.pkl
mv README.md ../../../profiles/README_4_3.md