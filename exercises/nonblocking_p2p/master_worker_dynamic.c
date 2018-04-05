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
        int data_count = 0;     /* first data is empty */
        int task_id = 0;
        while (1) {
            /* send data to master */
            MPI_Send(data, data_count, MPI_INT, size - 1, task_id, MPI_COMM_WORLD);
            /* receive data from master */
            MPI_Recv(data, 100, MPI_INT, size - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &data_count);
            if (data_count == 0) {
                /* received termination request */
                break;
            } else {
                /* received work */
                task_id = status.MPI_TAG;
                sleep(1);       /* compute data */
                /* send data to master in the next iteration */
            }
        }
    } else {
        /* master process */
        /* send data to workers. work size is 1 */
        int nprocessed = 0;     /* number of processed work units */
        int nterminated = 0;    /* number of terminated workers */
        int recv_buffer[100];

        MPI_Request reqs[5];
        MPI_Request *p_recv_req = &reqs[0];     /* 1st req is for recv */
        MPI_Request *send_reqs = &reqs[1];      /* the other 4 reqs (2nd to 5th) are for send */
        for (i = 0; i < 5; i++) {
            reqs[i] = MPI_REQUEST_NULL;
        }
        while (nterminated < size - 1) {
            int task_id;
            int data_count;
            int index;
            if (*p_recv_req == MPI_REQUEST_NULL) {
                /* receive request from worker */
                MPI_Irecv(recv_buffer, 100, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                          p_recv_req);
            }
            MPI_Waitany(5, reqs, &index, &status);
            if (index == 0) {
                /* received message */
                MPI_Get_count(&status, MPI_INT, &count);
                if (count != 0) {
                    /* show result computed by worker */
                    printf("worker ID: %d; task ID: %d; count: %d\n", status.MPI_SOURCE,
                           status.MPI_TAG, count);
                }
                /* find an empty slot for future Isend */
                for (index = 0; index < 4; index++) {
                    if (send_reqs[index] == MPI_REQUEST_NULL)
                        break;
                }
                if (index == 4) /* no slot is available */
                    MPI_Waitany(4, send_reqs, &index, MPI_STATUS_IGNORE);
                /* send_reqs[index] is MPI_REQUEST_NULL */
                if (nprocessed < 100) {
                    /* send work to worker */
                    task_id = nprocessed;
                    data_count = 1 + rand() % 4;
                    if (nprocessed + data_count > 100)
                        data_count = 100 - nprocessed;
                    MPI_Isend(data + nprocessed, data_count, MPI_INT, status.MPI_SOURCE, task_id,
                              MPI_COMM_WORLD, &send_reqs[index]);
                    nprocessed += data_count;
                } else {
                    /* send termination request to worker because all work has been processed */
                    MPI_Isend(NULL, 0, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD,
                              &send_reqs[index]);
                    nterminated++;
                }
            } else {
                /* sent message successfully */
            }
        }
        /* Be sure to complete all send_reqs */
        MPI_Waitall(4, send_reqs, MPI_STATUSES_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
