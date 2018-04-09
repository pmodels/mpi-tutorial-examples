MPICXX=mpicxx
MPICC=mpicc
OBJS=../../common/stencil/printarr_par.o
CFLAGS= -g -Wall -I../../common/stencil
BSPMM_OBJS=$(PWD)/../../common/bspmm/bspmm_common.o
BSPMM_CFLAGS=$(CFLAGS) -I$(PWD)/../../common/bspmm/
BINS=stencil_funneled stencil_multiple stencil_multiple_ncomms bspmm_funneled

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

stencil_%: stencil_%.c $(OBJS)
	$(MPICC) $(CFLAGS) -fopenmp -o $@ $< $(OBJS) -lm

bspmm_funneled: bspmm_funneled.c $(BSPMM_OBJS)
	$(MPICC) $(BSPMM_CFLAGS) -fopenmp -o $@ $< $(BSPMM_OBJS)

clean:
	rm -f $(BINS) $(OBJS)
	rm -fr output*.bmp
