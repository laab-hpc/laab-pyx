# Require LAAB_BUILD_DIR
ifndef LAAB_BUILD_DIR
$(error LAAB_BUILD_DIR is not set. Please export it before running make.)
endif

#LAAB_PREFIX = OMP_NUM_THREADS=58
# LAAB_REPS = 5
# LAAB_N = 3000
# LD_BLAS = -lopenblas
# LD_BLAS = -lmkl_rt

$(shell mkdir -p $(LAAB_BUILD_DIR))

stridiagmm:
	gcc -O3 stridiagmm.c -c -fopenmp -o $(LAAB_BUILD_DIR)/stridiagmm.o
	gcc $(LAAB_BUILD_DIR)/stridiagmm.o -o $(LAAB_BUILD_DIR)/stridiagmm.exe -fopenmp

run: stridiagmm
	REP=$(LAAB_REPS) $(LAAB_PREFIX) $(LAAB_BUILD_DIR)/stridiagmm.exe $(LAAB_N) $(LAAB_N)
	rm -f $(LAAB_BUILD_DIR)/*.o $(LAAB_BUILD_DIR)/*.exe

clean:
	rm -f $(LAAB_BUILD_DIR)/*.o $(LAAB_BUILD_DIR)/*.exe
