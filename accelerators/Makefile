CC=mpicc
STENCIL_COMMON_SRC=../common/stencil/printarr_par.c
STENCIL_CFLAGS= -acc -g -Wall -I../common/stencil
BINS=stencil

all: $(BINS)

stencil: stencil.c $(STENCIL_COMMON_SRC)
	$(CC) $(STENCIL_CFLAGS) $^ -o $@ -lm

clean:
	rm -f $(BINS)
	rm -f output*.bmp
