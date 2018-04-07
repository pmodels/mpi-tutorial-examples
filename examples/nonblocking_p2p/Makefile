MPICXX=mpicxx
MPICC=mpicc
OBJS=$(PWD)/../../common/stencil/printarr_par.o
CFLAGS= -g -Wall -I$(PWD)/../../common/stencil
BINS=stencil

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

stencil: stencil.c $(OBJS)
	$(MPICC) $(CFLAGS) -o $@ $< $(OBJS) -lm

clean:
	rm -f $(BINS) $(OBJS)
	rm -fr output*.bmp
