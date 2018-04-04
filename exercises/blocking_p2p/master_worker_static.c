#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    int rank, i, count, size;
    int data[100];
    MPI_Status status;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL));

    /* all processes except the last one are workers */
    if (rank < size - 1) {
        int task_id;
        int data_count;
        /* receive data from master */
        MPI_Recv(data, 100, MPI_INT, size - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        task_id = status.MPI_TAG;
        MPI_Get_count(&status, MPI_INT, &data_count);
        /* send data to master */
        sleep(1);       /* compute data */
        MPI_Send(data, data_count, MPI_INT, size - 1, task_id, MPI_COMM_WORLD);
    } else {
        /* master process */
        /* send data to workers */
        for (i = 0; i < size - 1; i++) {
            int task_id = i / 3;        /* three processes per task ID */
            int data_count = rand() % 100;      /* set work size */
            MPI_Send(data, data_count, MPI_INT, i, task_id, MPI_COMM_WORLD);
        }
        /* receive data from workers */
        for (i = 0; i < size - 1; i++) {
            MPI_Recv(data, 100, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &count);
            printf("worker ID: %d; task ID: %d; count: %d\n", status.MPI_SOURCE, status.MPI_TAG,
                   count);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
