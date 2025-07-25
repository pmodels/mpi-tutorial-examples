#include <omp.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    MPI_Init(0, 0);

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
    int tag = 0;

    void *buf = malloc(count);

    #pragma omp parallel num_threads(8)
    {
        int thread_id = omp_get_thread_num();
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

    MPI_Finalize();
    return 0;
}
