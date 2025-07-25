#include <omp.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 8

int main(void)
{
    // 1. use MPI_Init_thread
    int provided;
    MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        printf("MPI_THREAD_MULTIPLE is not supported\n");
        exit(0);
    }

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 2) {
        if (rank == 0) {
            printf("Launch with 2 processes!\n");
        }
        return 1;
    }

    int count = 1024;

    // 2. avoid race condition on message buffers
    void *bufs[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        bufs[i] = malloc(count);
    }

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int thread_id = omp_get_thread_num();
        void *buf = bufs[thread_id];
        // 3. use tag to control message matching
        int tag = thread_id;
        if (rank == 0) {
            int *p = buf;
            p[0] = thread_id;
            MPI_Send(buf, count, MPI_BYTE, 1, tag, comm);
        } else {
            MPI_Recv(buf, count, MPI_BYTE, 0, tag, comm, MPI_STATUS_IGNORE);
            int *p = buf;
            printf("Thread %d received from thread %d\n", thread_id, p[0]);
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        free(bufs[i]);
    }
    MPI_Finalize();
    return 0;
}
