#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    int rank, size;
    int color, split_rank, split_size;
    float *sendbuf, *recvbuf;
    MPI_Comm split_comm;
    MPI_Request req;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* color evens and odds */
    color = (rank % 2 == 0);
    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &split_comm);
    if (color == 1)
        MPI_Comm_set_name(split_comm, "Even Comm");
    else
        MPI_Comm_set_name(split_comm, "Odd Comm");

    MPI_Comm_rank(split_comm, &split_rank);
    MPI_Comm_size(split_comm, &split_size);

    if (color == 1 && split_rank == 0)
        MPI_Irecv(NULL, 0, MPI_INT, 1, 0, MPI_COMM_WORLD, &req);

    if (color == 0) {
        int i, curr, num_ops = (split_size - 1) * 2;
        MPI_Request *reqs;

        sendbuf = malloc(sizeof(float) * split_size);
        recvbuf = malloc(sizeof(float) * split_size);
        reqs = malloc(sizeof(MPI_Request) * num_ops);

        for (i = 0, curr = 0; i < split_size; i++) {
            if (i != split_rank)
                MPI_Irecv(&recvbuf[i], 1, MPI_FLOAT, i, 0, split_comm, &reqs[curr++]);
        }

        for (i = 0; i < split_size; i++) {
            sleep(1);
            if (i != split_rank)
                MPI_Isend(&sendbuf[i], 1, MPI_FLOAT, i, 0, split_comm, &reqs[curr++]);
        }

        MPI_Waitall(num_ops, reqs, MPI_STATUSES_IGNORE);

        if (split_rank == 0)
            MPI_Isend(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD, &req);
    }

    if (split_rank == 0)
        MPI_Wait(&req, MPI_STATUS_IGNORE);

    if (color == 1)
        MPI_Barrier(split_comm);

    MPI_Finalize();
    return 0;
}
