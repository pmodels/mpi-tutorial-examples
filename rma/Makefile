MPICXX=mpicxx
MPICC=mpicc
OBJS=$(PWD)/../../common/stencil/printarr_par.o
CFLAGS= -g -Wall -I$(PWD)/../../common/stencil
BSPMM_OBJS=$(PWD)/../../common/bspmm/bspmm_common.o
BSPMM_CFLAGS=$(CFLAGS) -I$(PWD)/../../common/bspmm/
BINS=stencil_fence_put stencil_fence_get stencil_pscw_put stencil_lock_put
BINS+=bspmm bspmm_counter

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

stencil_%: stencil_%.c $(OBJS)
	$(MPICC) $(CFLAGS) -o $@ $< $(OBJS) -lm

bspmm: bspmm.c $(BSPMM_OBJS)
	$(MPICC) $(BSPMM_CFLAGS) -o $@ $< $(BSPMM_OBJS)

bspmm_counter: bspmm_counter.c $(BSPMM_OBJS)
	$(MPICC) $(BSPMM_CFLAGS) -o $@ $< $(BSPMM_OBJS)

clean:
	rm -f $(BINS) $(OBJS)
	rm -fr output*.bmp
