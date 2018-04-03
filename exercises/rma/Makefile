MPICXX=mpicxx
MPICC=mpicc
OBJS=$(PWD)/../../common/stencil/printarr_par.o
CFLAGS= -g -Wall -I$(PWD)/../../common/stencil
BINS=stencil_fence_put stencil_fence_get stencil_pscw_put stencil_lock_put

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

stencil_%: stencil_%.c $(OBJS)
	$(MPICC) $(CFLAGS) -o $@ $< $(OBJS) -lm

clean:
	rm -f $(BINS) $(OBJS)
	rm -fr output*.bmp
