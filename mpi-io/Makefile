CC=mpicc
STENCIL_COMMON_SRC=../common/stencil/printarr_par.c
STENCIL_CFLAGS= -g -I../common/stencil
BINS=stencil_carttopo_checkpoint_stdout stencil_carttopo_checkpoint_mpiio

all: $(BINS)

stencil_carttopo_checkpoint_stdout: stencil_carttopo_checkpoint.c checkpoint_stdout.c $(STENCIL_COMMON_SRC)
	$(CC) $(STENCIL_CFLAGS) -o $@ $^

stencil_carttopo_checkpoint_mpiio: stencil_carttopo_checkpoint.c checkpoint_mpiio.c $(STENCIL_COMMON_SRC)
	$(CC) $(STENCIL_CFLAGS) -o $@ $^

clean:
	rm -f $(BINS)
	rm -fr output*.bmp *.dSYM
	rm -f *.o *.chkpt
