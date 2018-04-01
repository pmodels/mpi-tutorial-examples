MPICXX=mpicxx
MPICC=mpicc
OBJS=../printarr_par.o
CFLAGS= -g -Wall -I../
BINS=ddt_thread_funneled ddt_thread_multiple

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

ddt_thread_%: ddt_thread_%.c $(OBJS)
	$(MPICC) $(CFLAGS) -fopenmp -o $@ $< $(OBJS) -lm

clean:
	rm -f $(BINS) $(OBJS)
	rm -fr output*.bmp
