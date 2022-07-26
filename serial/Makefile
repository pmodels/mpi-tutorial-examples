# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

CC=gcc
CFLAGS= -g -Wall
BINS=stencil

all: $(BINS)

stencil: stencil.c ../common/stencil/printarr.c
	$(CC) $(STENCIL_CFLAGS) $^ -o $@ -lm

clean:
	rm -f $(BINS)
	rm -f output*tmp
