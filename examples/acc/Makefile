MPICC=mpicc
OBJS=$(PWD)/../../common/stencil/printarr_par.o
CFLAGS= -acc -g -Wall -I$(PWD)/../../common/stencil
BINS=stencil_mpi_acc_update

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

stencil_mpi_acc_update: stencil_mpi_acc_update.c $(OBJS)
	$(MPICC) $(CFLAGS) -o $@ $< $(OBJS) -lm

clean:
	rm -f $(BINS) $(OBJS)
