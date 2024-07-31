# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

ifdef CRAY_CPU_TARGET
CC=cc
else
CC=mpicc
endif
CFLAGS= -g -Wall -I../common
BINS=stencil stencil_complete
STENCIL_DEPS=../common/stencil/printarr.c ../common/perf_stat.c

all: $(BINS)

stencil: stencil.c
	$(CC) stencil.c -o stencil

stencil_complete: stencil_complete.c $(STENCIL_DEPS)
	$(CC) $(CFLAGS) $^ -o $@ -lm

clean:
	rm -f $(BINS)
	rm -f output*tmp
