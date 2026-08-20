#!/bin/bash

source /proj/nobackup/aravind/LAAB-HPC/benchmarks/laab-pyx/venvs/pyx-kebnekaise/activate.sh

mkdir -p ../../../profiles

pyx-process-logs data.txt --outfile profile.pkl --name "CPU-4x" --system "kebnekaise:skylake" \
    --lib "jax" --version "0.4.25" --toolchain "gfbf-2023a" \
    --config_path ../../src/config.json --prob_siz "3000" --reps "10" --omp_num_threads "4"

pyx-prepare-md-report profile.pkl --outfile README.md

mv profile.pkl ../../../profiles/profile_4_0.pkl
mv README.md ../../../profiles/README_4_0.md