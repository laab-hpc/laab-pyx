ml GCC FlexiBLAS
export LAAB_BUILD_DIR=build
LD_BLAS=-lopenblas make
./build/sgemv.exe 1000 1000 1000
