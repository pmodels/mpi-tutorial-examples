MPICXX=mpicxx
MPICC=mpicc
OBJS=$(PWD)/../../common/stencil/printarr_par.o
CFLAGS= -g -Wall -I$(PWD)/../../common/stencil
BINS=stencil simple_dbg

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

stencil: stencil.c $(OBJS)
	$(MPICC) $(CFLAGS) -o $@ $< $(OBJS) -lm

simple_dbg: simple_dbg.c
	$(MPICC) -O0 -g -o $@ $<

clean:
	rm -f $(BINS) $(OBJS)
	rm -fr output*.bmp
