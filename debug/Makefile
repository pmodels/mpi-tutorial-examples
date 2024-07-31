# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

ifdef CRAY_CPU_TARGET
CC=cc
else
CC=mpicc
endif
CFLAGS= -O0 -g -Wall
BINS=simple deadlock split

all: $(BINS)

clean:
	rm -f $(BINS)
