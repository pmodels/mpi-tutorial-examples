/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <stdio.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
    int rank;
    int size;

    MPI_Init(0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Hello from process %d of %d\n", rank, size);
    MPI_Finalize();
    return 0;
}
