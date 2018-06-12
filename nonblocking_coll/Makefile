# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

CC=mpicc
CFLAGS= -g -Wall
STENCIL_COMMON_SRC=../common/stencil/printarr_par.c
STENCIL_CFLAGS=$(CFLAGS) -I../common/stencil
BINS=stencil_alltoallw

all: $(BINS)

stencil_alltoallw: stencil_alltoallw.c $(STENCIL_COMMON_SRC)
	$(CC) $(STENCIL_CFLAGS) $^ -o $@ -lm

clean:
	rm -f $(BINS)
