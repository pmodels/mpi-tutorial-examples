CC=mpicc
CFLAGS= -g -Wall
STENCIL_COMMON_SRC=../common/stencil/printarr_par.c
STENCIL_CFLAGS= $(CFLAGS) -I../common/stencil
BINS=cpi stencil_alltoallv sort_any_procs

all: $(BINS)

sort_any_procs: sort_any_procs.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

stencil_alltoallv: stencil_alltoallv.c $(STENCIL_COMMON_SRC)
	$(CC) $(STENCIL_CFLAGS) $^ -o $@ -lm

clean:
	rm -f $(BINS)
	rm -f output*.bmp
