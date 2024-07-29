# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

CC=gcc
CFLAGS= -g -Wall -I../common
BINS=stencil_complete
STENCIL_DEPS=../common/stencil/printarr.c ../common/perf_stat.c

all: $(BINS)

stencil_complete: stencil_complete.c $(STENCIL_DEPS)
	$(CC) $(CFLAGS) $^ -o $@ -lm

clean:
	rm -f $(BINS)
	rm -f output*tmp
