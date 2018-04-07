MPICXX=mpicxx
MPICC=mpicc
CFLAGS= -g -Wall
BSPMM_OBJS=$(PWD)/../../common/bspmm/bspmm_common.o
BSPMM_CFLAGS=$(CFLAGS) -I$(PWD)/../../common/bspmm/
BINS=sort_any_procs bspmm_blocking bspmm_blocking_getcount

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

sort_any_procs: sort_any_procs.c
	$(MPICC) $(CFLAGS) $^ -o $@ -lm

bspmm_blocking: bspmm_blocking.c $(BSPMM_OBJS)
	$(MPICC) $(BSPMM_CFLAGS) -o $@ $< $(BSPMM_OBJS)

bspmm_blocking_getcount: bspmm_blocking_getcount.c $(BSPMM_OBJS)
	$(MPICC) $(BSPMM_CFLAGS) -o $@ $< $(BSPMM_OBJS)

clean:
	rm -f $(BINS) $(BSPMM_OBJS)
