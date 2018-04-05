MPICXX=mpicxx
MPICC=mpicc
CFLAGS= -g -Wall
BINS=sendrecv master_worker_simple sort_2_procs

all: $(BINS)

sendrecv: sendrecv.c
	$(MPICC) $(CFLAGS) $^ -o $@

master_worker_simple: master_worker_static.c
	$(MPICC) $(CFLAGS) $^ -o $@

sort_2_procs: sort_2_procs.c
	$(MPICC) $(CFLAGS) $^ -o $@

clean:
	rm -f $(BINS)
