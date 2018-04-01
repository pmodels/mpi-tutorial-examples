MPICXX=mpicxx
MPICC=mpicc
OBJS=../printarr_par.o
CFLAGS= -g -Wall -I../
BINS=ddt_rma_fence_get ddt_rma_fence_put ddt_rma_lock_put ddt_rma_pscw_put

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

ddt_rma_%: ddt_rma_%.c $(OBJS)
	$(MPICC) $(CFLAGS) -o $@ $< $(OBJS) -lm

clean:
	rm -f $(BINS) $(OBJS)
	rm -fr output*.bmp
