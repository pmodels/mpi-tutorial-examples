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

    int count = 1000000000;
    int tag = 0;
    int repeat = 2;

    if (rank == 0) {
        printf("Measuring bandwidth with message size %d bytes repeat %d times\n", count, repeat);
    }

    void *buf = malloc(count);

    for (int round = 0; round < 5; round++) {
        if (rank == 0) {
            printf("Round %d\n", round);
            double time_start = MPI_Wtime();
            for (int i = 0; i < repeat; i++) {
                MPI_Send(buf, count, MPI_BYTE, 1, tag, comm);
                MPI_Recv(NULL, 0, MPI_DATATYPE_NULL, 1, tag, comm, MPI_STATUS_IGNORE);
            }
            double time_finish = MPI_Wtime();
            double time_dur = (time_finish - time_start) / repeat;
            printf("    Average bandwidth: %f GB/sec\n", count / time_dur / 1e9);
        } else {
            for (int i = 0; i < repeat; i++) {
                MPI_Recv(buf, count, MPI_BYTE, 0, tag, comm, MPI_STATUS_IGNORE);
                MPI_Send(NULL, 0, MPI_DATATYPE_NULL, 0, tag, comm);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
