CC=mpicc
CFLAGS= -g -Wall
BSPMM_COMMON_SRC=../common/bspmm/bspmm_common.c
BSPMM_CFLAGS=$(CFLAGS) -I../common/bspmm
BINS=sendrecv master_worker_simple master_worker sort_2_procs
BINS+=sort_2_procs sort_any_procs bspmm_simple bspmm_pack

all: $(BINS)

sort_2_procs: sort_2_procs.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

sort_any_procs: sort_any_procs.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

bspmm_simple: bspmm_simple.c $(BSPMM_COMMON_SRC)
	$(CC) $(BSPMM_CFLAGS) $^ -o $@

bspmm_pack: bspmm_pack.c $(BSPMM_COMMON_SRC)
	$(CC) $(BSPMM_CFLAGS) $^ -o $@

clean:
	rm -f $(BINS)
