#!/bin/bash

source /proj/nobackup/aravind/LAAB-HPC/benchmarks/laab-pyx/venvs/pyx-kebnekaise/activate.sh

mkdir -p ../../../profiles

pyx-process-logs data.txt --outfile profile.pkl --name "CPU-14x" --system "kebnekaise:skylake" \
    --lib "jax" --version "0.4.25" --toolchain "gfbf-2023a" \
    --config_path ../../src/config.json --prob_siz "3000" --reps "10" --omp_num_threads "14"

pyx-prepare-md-report profile.pkl --outfile README.md

mv profile.pkl ../../../profiles/profile_14_0.pkl
mv README.md ../../../profiles/README_14_0.md