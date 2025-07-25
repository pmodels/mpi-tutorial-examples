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

    void *buf;
    if (rank == 0) {
        buf = omp_target_alloc(count, 0);
    } else {
        buf = omp_target_alloc(count, 1);
    }

    if (rank == 0) {
        MPI_Send(buf, count, MPI_BYTE, 1, tag, comm);
    } else {
        MPI_Recv(buf, count, MPI_BYTE, 0, tag, comm, MPI_STATUS_IGNORE);
    }

    MPI_Finalize();
    return 0;
}
