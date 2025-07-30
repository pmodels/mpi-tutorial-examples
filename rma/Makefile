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
STENCIL_COMMON_SRC=../common/stencil/stencil_par.c ../common/stencil/printarr_par.c ../common/perf_stat.c
STENCIL_CFLAGS=$(CFLAGS) -I../common/stencil
BSPMM_COMMON_SRC=../common/bspmm/bspmm_common.c
BSPMM_CFLAGS=$(CFLAGS) -I../common/bspmm
DISPW_CFLAGS=$(CFLAGS)
BINS=stencil_fence_put stencil_fence_get stencil_pscw_put stencil_lock_put cpi cpi_task1 cpi_task2 cpi_task3
BINS+=bspmm bspmm_counter
BINS+=dispatch_worker rma_dispatch_worker

all: $(BINS)

stencil_fence_put: stencil_fence_put.c $(STENCIL_COMMON_SRC)
	$(CC) $(STENCIL_CFLAGS) $^ -o $@ -lm

stencil_fence_get: stencil_fence_get.c $(STENCIL_COMMON_SRC)
	$(CC) $(STENCIL_CFLAGS) $^ -o $@ -lm

stencil_pscw_put: stencil_pscw_put.c $(STENCIL_COMMON_SRC)
	$(CC) $(STENCIL_CFLAGS) $^ -o $@ -lm

stencil_lock_put: stencil_lock_put.c $(STENCIL_COMMON_SRC)
	$(CC) $(STENCIL_CFLAGS) $^ -o $@ -lm

bspmm: bspmm.c $(BSPMM_COMMON_SRC)
	$(CC) $(BSPMM_CFLAGS) $^ -o $@

bspmm_counter: bspmm_counter.c $(BSPMM_COMMON_SRC)
	$(CC) $(BSPMM_CFLAGS) $^ -o $@

dispatch_worker: dispatch_worker.c
	$(CC) $(DISPW_CFLAGS) $^ -o $@

rma_dispatch_worker: rma_dispatch_worker.c
	$(CC) $(DISPW_CFLAGS) $^ -o $@

cpi: cpi.c
	$(CC) $^ -o $@

cpi_task1: cpi_task1.c
	$(CC) $^ -o $@

cpi_task2: cpi_task2.c
	$(CC) $^ -o $@

cpi_task3: cpi_task3.c
	$(CC) $^ -o $@

clean:
	rm -f $(BINS)
	rm -f output*.bmp
