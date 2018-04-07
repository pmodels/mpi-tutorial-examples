MPICXX=mpicxx
MPICC=mpicc
CFLAGS= -g -Wall
STENCIL_OBJS=$(PWD)/../../common/stencil/printarr_par.o
STENCIL_CFLAGS= $(CFLAGS) -I$(PWD)/../../common/stencil
BINS=stencil sort_any_procs

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

sort_any_procs: sort_any_procs.c
	$(MPICC) $(CFLAGS) $^ -o $@

stencil: stencil.c $(STENCIL_OBJS)
	$(MPICC) $(STENCIL_CFLAGS) -o $@ $< $(STENCIL_OBJS) -lm

clean:
	rm -f $(BINS) $(STENCIL_OBJS)
	rm -fr output*.bmp
