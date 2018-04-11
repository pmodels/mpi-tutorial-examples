MPICXX=mpicxx
MPICC=mpicc
CFLAGS= -g -Wall
STENCIL_OBJS=$(PWD)/../../common/stencil/printarr_par.o
STENCIL_CFLAGS= -g -Wall -I$(PWD)/../../common/stencil
BSPMM_OBJS=$(PWD)/../../common/bspmm/bspmm_common.o
BSPMM_CFLAGS=$(CFLAGS) -I$(PWD)/../../common/bspmm/
BINS=stencil bspmm_pack

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

stencil: stencil.c $(STENCIL_OBJS)
	$(MPICC) $(STENCIL_CFLAGS) -o $@ $< -lm

bspmm_pack: bspmm_pack.c $(BSPMM_OBJS)
	$(MPICC) $(BSPMM_CFLAGS) -o $@ $< $(BSPMM_OBJS)

clean:
	rm -f $(BINS) $(BSPMM_OBJS)
