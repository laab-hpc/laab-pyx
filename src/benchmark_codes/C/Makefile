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

sgemv:
	gcc -O3 sgemv.c -c -o $(LAAB_BUILD_DIR)/sgemv.o
	gcc $(LAAB_BUILD_DIR)/sgemv.o -o $(LAAB_BUILD_DIR)/sgemv.exe $(LD_BLAS)

ssyrk:
	gcc -O3 ssyrk.c -c -o $(LAAB_BUILD_DIR)/ssyrk.o
	gcc $(LAAB_BUILD_DIR)/ssyrk.o -o $(LAAB_BUILD_DIR)/ssyrk.exe $(LD_BLAS)

strmm:
	gcc -O3 strmm.c -c -o $(LAAB_BUILD_DIR)/strmm.o
	gcc $(LAAB_BUILD_DIR)/strmm.o -o $(LAAB_BUILD_DIR)/strmm.exe $(LD_BLAS)

stridiagmm:
	gcc -O3 stridiagmm.c -c -fopenmp -o $(LAAB_BUILD_DIR)/stridiagmm.o
	gcc $(LAAB_BUILD_DIR)/stridiagmm.o -o $(LAAB_BUILD_DIR)/stridiagmm.exe -fopenmp

run: sgemm sgemv ssyrk strmm stridiagmm
	REP=$(LAAB_REPS) $(LAAB_PREFIX) $(LAAB_BUILD_DIR)/sgemm.exe $(LAAB_N) $(LAAB_N) $(LAAB_N)
	REP=$(LAAB_REPS) $(LAAB_PREFIX) $(LAAB_BUILD_DIR)/sgemv.exe $(LAAB_N) $(LAAB_N) $(LAAB_N)
	REP=$(LAAB_REPS) $(LAAB_PREFIX) $(LAAB_BUILD_DIR)/ssyrk.exe $(LAAB_N) $(LAAB_N)
	REP=$(LAAB_REPS) $(LAAB_PREFIX) $(LAAB_BUILD_DIR)/strmm.exe $(LAAB_N) $(LAAB_N)
	REP=$(LAAB_REPS) $(LAAB_PREFIX) $(LAAB_BUILD_DIR)/stridiagmm.exe $(LAAB_N) $(LAAB_N)
	rm -f $(LAAB_BUILD_DIR)/*.o $(LAAB_BUILD_DIR)/*.exe

clean:
	rm -f $(LAAB_BUILD_DIR)/*.o $(LAAB_BUILD_DIR)/*.exe
