MPICXX=mpicxx
MPICC=mpicc
CFLAGS= -g -Wall
BINS=master_worker_static master_worker_dynamic sort_2_procs sort_any_procs

all: $(BINS)

sort_2_procs: sort_2_procs.c
	$(MPICC) $(CFLAGS) $^ -o $@

sort_any_procs: sort_any_procs.c
	$(MPICC) $(CFLAGS) $^ -o $@

master_worker_static: master_worker_static.c
	$(MPICC) $(CFLAGS) $^ -o $@

master_worker_dynamic: master_worker_dynamic.c
	$(MPICC) $(CFLAGS) $^ -o $@

clean:
	rm -f $(BINS)
