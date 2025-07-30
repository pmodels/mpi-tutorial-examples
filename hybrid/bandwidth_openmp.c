#include <omp.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 8

int main(void)
{
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

    int count_total = 1000000000;
    int count = count_total / NUM_THREADS;
    int repeat = 2;

    void *bufs[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        bufs[i] = malloc(count);
    }

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int thread_id = omp_get_thread_num();
        void *buf = bufs[thread_id];
        int tag = thread_id;

        for (int round = 0; round < 5; round++) {
            if (rank == 0) {
                double time_start = MPI_Wtime();
                for (int i = 0; i < repeat; i++) {
                    MPI_Send(buf, count, MPI_BYTE, 1, tag, comm);
                    MPI_Recv(NULL, 0, MPI_DATATYPE_NULL, 1, tag, comm, MPI_STATUS_IGNORE);
                }
                double time_dur = MPI_Wtime() - time_start;
                printf("Thread %d Round %d, Average bandwidth: %f GB/sec, count=%d, dur=%f\n", thread_id, round, (count / 1e9) / (time_dur / repeat), count, time_dur);
            } else {
                for (int i = 0; i < repeat; i++) {
                    MPI_Recv(buf, count, MPI_BYTE, 0, tag, comm, MPI_STATUS_IGNORE);
                    MPI_Send(NULL, 0, MPI_DATATYPE_NULL, 0, tag, comm);
                }
            }
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        free(bufs[i]);
    }
    MPI_Finalize();
    return 0;
}
