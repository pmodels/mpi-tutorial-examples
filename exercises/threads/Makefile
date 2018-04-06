MPICXX=mpicxx
MPICC=mpicc
OBJS=../../common/stencil/printarr_par.o
CFLAGS= -g -Wall -I../../common/stencil
BINS=stencil_funneled stencil_multiple_ncomms

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

stencil_%: stencil_%.c $(OBJS)
	$(MPICC) $(CFLAGS) -fopenmp -o $@ $< $(OBJS) -lm

clean:
	rm -f $(BINS) $(OBJS)
	rm -fr output*.bmp
