MPICXX=mpicxx
MPICC=mpicc
CFLAGS= -g
BINS=stencil_mpi_carttopo_stdout stencil_mpi_carttopo_mpiio

all: $(BINS)

stencil_mpi_carttopo_stdout: stencil_mpi_carttopo.c stencil_io_stdout.c printarr_par.c
	$(MPICC) $(CFLAGS) -o $@ $^

stencil_mpi_carttopo_mpiio: stencil_mpi_carttopo.c stencil_io_mpiio.c printarr_par.c
	$(MPICC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(BINS)
	rm -fr output*.bmp *.dSYM
	rm -f *.o *.chkpt
