# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

ifdef CRAY_CPU_TARGET
CC=cc
else
CC=mpicc
endif
CFLAGS= -g -Wall
BINS=hello_world hello_session

all: $(BINS)

hello_world: hello_world.c
	$(CC) $(CFLAGS) $^ -o $@

hello_session: hello_session.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f $(BINS)
