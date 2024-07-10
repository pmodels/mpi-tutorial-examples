/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    int rank, data[5];
    MPI_Datatype dtype;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Type_vector(3, 1, 2, MPI_INT, &dtype);
    MPI_Type_commit(&dtype);

    if (rank == 0) {
        data[0] = 0;
        data[1] = 10;
        data[2] = 20;
        data[3] = 30;
        data[4] = 40;
        MPI_Send(data, 1, dtype, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        data[0] = -1;
        data[1] = -1;
        data[2] = -1;
        data[3] = -1;
        data[4] = -1;
        printf("[%d] before receiving: %d,%d,%d,%d,%d\n", rank,
               data[0], data[1], data[2], data[3], data[4]);
        MPI_Recv(data, 1, dtype, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[%d] after receiving: %d,%d,%d,%d,%d\n", rank,
               data[0], data[1], data[2], data[3], data[4]);
    }

    MPI_Type_free(&dtype);

    MPI_Finalize();
    return 0;
}
