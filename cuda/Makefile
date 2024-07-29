# -*- Mode: Makefile; -*-
#
# See COPYRIGHT in top-level directory.
#

NVCC=nvcc
BINS=stencil_cuda stencil_mpi_cuda

all: $(BINS)

stencil_cuda: stencil_cuda.cu
	$(NVCC) stencil_cuda.cu -o stencil_cuda -lmpi

stencil_mpi_cuda: stencil_mpi_cuda.cu
	$(NVCC) stencil_mpi_cuda.cu -o stencil_mpi_cuda -lmpi

clean:
	rm -f $(BINS)
