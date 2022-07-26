# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

CFLAGS= -g -Wall
BINS=stencil

all: $(BINS)

stencil: stencil.c ../common/stencil/printarr.c

clean:
	rm -f $(BINS)
	rm -f output*tmp
