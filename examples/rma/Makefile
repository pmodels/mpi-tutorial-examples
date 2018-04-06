MPICXX=mpicxx
MPICC=mpicc
OBJS=$(PWD)/../../common/bspmm/bspmm_common.o
CFLAGS= -g -Wall -I$(PWD)/../../common/bspmm
BINS=bspmm

all: $(BINS)

%.o: %.c
	$(MPICC) $(CFLAGS) $< -c -lm -o $@

bspmm: bspmm.c $(OBJS)
	$(MPICC) $(CFLAGS) -o $@ $< $(OBJS) -lm

clean:
	rm -f $(BINS) $(OBJS)
