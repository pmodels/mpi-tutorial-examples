MPICXX=mpicxx
MPICC=mpicc
CFLAGS= -g -Wall
BSPMM_OBJS=$(PWD)/../../common/bspmm/bspmm_common.o
BSPMM_CFLAGS=$(CFLAGS) -I$(PWD)/../../common/bspmm/
BINS=sendrecv master_worker_simple sort_2_procs bspmm_simple

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

sendrecv: sendrecv.c
	$(MPICC) $(CFLAGS) $^ -o $@

master_worker_simple: master_worker_simple.c
	$(MPICC) $(CFLAGS) $^ -o $@

sort_2_procs: sort_2_procs.c
	$(MPICC) $(CFLAGS) $^ -o $@

bspmm_simple: bspmm_simple.c $(BSPMM_OBJS)
	$(MPICC) $(BSPMM_CFLAGS) -o $@ $< $(BSPMM_OBJS)

clean:
	rm -f $(BINS) $(BSPMM_OBJS)
