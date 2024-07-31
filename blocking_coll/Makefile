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
STENCIL_COMMON_SRC=../common/stencil/printarr_par.c ../common/stencil/stencil_par.c ../common/perf_stat.c
STENCIL_CFLAGS= $(CFLAGS) -I../common/stencil
BINS=cpi stencil_alltoallv sort_any_procs

all: $(BINS)

cpi: cpi.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

sort_any_procs: sort_any_procs.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

stencil_alltoallv: stencil_alltoallv.c $(STENCIL_COMMON_SRC)
	$(CC) $(STENCIL_CFLAGS) $^ -o $@ -lm

clean:
	rm -f $(BINS)
	rm -f output*.bmp
