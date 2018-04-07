MPICXX=mpicxx
MPICC=mpicc
CFLAGS= -g -Wall
BSPMM_OBJS=$(PWD)/../../common/bspmm/bspmm_common.o
BSPMM_CFLAGS=$(CFLAGS) -I$(PWD)/../../common/bspmm/
BINS=sort_any_procs bspmm_pack bspmm_pack_getcount

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

sort_any_procs: sort_any_procs.c
	$(MPICC) $(CFLAGS) $^ -o $@ -lm

bspmm_pack: bspmm_pack.c $(BSPMM_OBJS)
	$(MPICC) $(BSPMM_CFLAGS) -o $@ $< $(BSPMM_OBJS)

bspmm_pack_getcount: bspmm_pack_getcount.c $(BSPMM_OBJS)
	$(MPICC) $(BSPMM_CFLAGS) -o $@ $< $(BSPMM_OBJS)

clean:
	rm -f $(BINS) $(BSPMM_OBJS)
