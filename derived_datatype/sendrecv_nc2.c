/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    int rank, data[5];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        int sendbuf[3];
        data[0] = 0;
        data[1] = 10;
        data[2] = 20;
        data[3] = 30;
        data[4] = 40;
        sendbuf[0] = data[0];
        sendbuf[1] = data[2];
        sendbuf[2] = data[4];
        MPI_Send(sendbuf, 3, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        int recvbuf[3];
        data[0] = -1;
        data[1] = -1;
        data[2] = -1;
        data[3] = -1;
        data[4] = -1;
        printf("[%d] before receiving: %d,%d,%d,%d,%d\n", rank,
               data[0], data[1], data[2], data[3], data[4]);
        MPI_Recv(recvbuf, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data[0] = recvbuf[0];
        data[2] = recvbuf[1];
        data[4] = recvbuf[2];
        printf("[%d] after receiving: %d,%d,%d,%d,%d\n", rank,
               data[0], data[1], data[2], data[3], data[4]);
    }

    MPI_Finalize();
    return 0;
}
