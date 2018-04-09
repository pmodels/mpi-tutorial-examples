MPICXX=mpicxx
MPICC=mpicc
STENCIL_OBJS=$(PWD)/../../common/stencil/printarr_par.o
STENCIL_CFLAGS= -g -Wall -I$(PWD)/../../common/stencil
BSPMM_OBJS=$(PWD)/../../common/bspmm/bspmm_common.o
BSPMM_CFLAGS=$(CFLAGS) -I$(PWD)/../../common/bspmm/
BINS=stencil bspmm_vector bspmm_subarray

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

stencil: stencil.c $(OBJS) $(STENCIL_OBJS)
	$(MPICC) $(STENCIL_CFLAGS) -o $@ $< $(STENCIL_OBJS) -lm

bspmm_vector: bspmm_vector.c $(BSPMM_OBJS)
	$(MPICC) $(BSPMM_CFLAGS) -o $@ $< $(BSPMM_OBJS)

bspmm_subarray: bspmm_subarray.c $(BSPMM_OBJS)
	$(MPICC) $(BSPMM_CFLAGS) -o $@ $< $(BSPMM_OBJS)
	
clean:
	rm -f $(BINS) $(BSPMM_OBJS) $(STENCIL_OBJS)
	rm -fr output*.bmp
