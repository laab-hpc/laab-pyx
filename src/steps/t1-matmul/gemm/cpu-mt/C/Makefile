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

sgemm:
	gcc -O3 sgemm.c -c -o $(LAAB_BUILD_DIR)/sgemm.o
	gcc $(LAAB_BUILD_DIR)/sgemm.o -o $(LAAB_BUILD_DIR)/sgemm.exe $(LD_BLAS)

run: sgemm
	REP=$(LAAB_REPS) $(LAAB_PREFIX) $(LAAB_BUILD_DIR)/sgemm.exe $(LAAB_N) $(LAAB_N) $(LAAB_N)
	rm -f $(LAAB_BUILD_DIR)/*.o $(LAAB_BUILD_DIR)/*.exe

clean:
	rm -f $(LAAB_BUILD_DIR)/*.o $(LAAB_BUILD_DIR)/*.exe
