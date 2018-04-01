MPICXX=mpicxx
MPICC=mpicc
OBJS=printarr_par.o
CFLAGS= -g -Wall
BINS=ddt

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

ddt: ddt.c $(OBJS)
	$(MPICC) $(CFLAGS) -o $@ $< $(OBJS) -lm

clean:
	rm -f $(BINS) $(OBJS)
	rm -fr output*.bmp
