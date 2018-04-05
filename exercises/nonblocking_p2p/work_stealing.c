#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

const int TAG_RESULT = 0;       /* send result to master */
const int TAG_STEAL = 1;        /* request victim to give its task */
const int TAG_MIGRATE = 2;      /* give task to thief */
const int TAG_STOP_STEAL = 3;   /* tell worker to stop stealing */
const int TAG_FINISH_STEAL = 3; /* notify completion of stealing */
const int TAG_TERMINATE = 4;    /* tell worker to stop receiving and exit while-loop */

int main(int argc, char **argv)
{
    int rank, i, count, size;
    int local_data[100];
    MPI_Status status;

    int nprocessed = 0;         /* number of processed work units globally */
    int nlocal_processed = 0;   /* number of processed work units locally */
    int local_data_index = 0;
    int local_data_len;
    int send_result_flag = 0;   /* send result if this flag is 1 */
    /* stop stealing if flag != 0. send completion of stealing if flag is 1 */
    int stop_stealing_flag = 0;

    MPI_Request reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
    MPI_Request *recv_req = &reqs[0];
    MPI_Request *send_req = &reqs[1];
    int recv_buffer[1], send_buffer[1];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL));

    /* rank N has approx. (100 * N / (0 + 2 + ... + size - 1)) elements */
    local_data_len =
        (100 * (rank * (rank + 1) / 2) / (size * (size - 1) / 2)) -
        (100 * ((rank - 1) * rank / 2) / (size * (size - 1) / 2));

    /* this process can always receive message to receive work stealing request */
    MPI_Irecv(recv_buffer, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, recv_req);
    while (1) {
        int index, flag;
        if (local_data_index < local_data_len) {
            /* compute 1 work unit */
            sleep(1);
            local_data_index++;
            nlocal_processed++;
            if (local_data_index == local_data_len)
                send_result_flag = 1;
        } else {
            /* try to steal work */
            if (*send_req == MPI_REQUEST_NULL && stop_stealing_flag == 0) {
                int victim = (rank + rand() % (size - 1)) % size;
                MPI_Isend(NULL, 0, MPI_INT, victim, TAG_STEAL, MPI_COMM_WORLD, send_req);
            }
        }
        MPI_Testany(2, reqs, &index, &flag, &status);
        if (flag == 1 && index == 0) {
            /* receive message from other processes */
            int tag = status.MPI_TAG;
            if (tag == TAG_RESULT) {
                /* show result. rank must be 0 */
                printf("worker ID: %d; count: %d\n", status.MPI_SOURCE, recv_buffer[0]);
                nprocessed += recv_buffer[0];
                if (nprocessed == 100) {
                    /* finish all the other workers */
                    int nfinished = 1;  /* master process has been finished */
                    /* tell workers to stop work stealing */
                    for (i = 1; i < size; i++)
                        MPI_Send(NULL, 0, MPI_INT, i, TAG_FINISH_STEAL, MPI_COMM_WORLD);
                    do {
                        MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                                 &status);
                    } while (!(status.MPI_TAG == TAG_FINISH_STEAL && ++nfinished == size));
                    /* tell workers to stop receiving messages and exit loop */
                    for (i = 1; i < size; i++)
                        MPI_Send(NULL, 0, MPI_INT, i, TAG_TERMINATE, MPI_COMM_WORLD);
                    break;
                }
            } else if (tag == TAG_STEAL) {
                /* sender needs a task */
                if (local_data_index < local_data_len && *send_req == MPI_REQUEST_NULL) {
                    /* send one work unit */
                    MPI_Isend(local_data + local_data_index, 1, MPI_INT, status.MPI_SOURCE,
                              TAG_MIGRATE, MPI_COMM_WORLD, send_req);
                    local_data_index++;
                    if (local_data_index == local_data_len)
                        send_result_flag = 1;
                }
            } else if (tag == TAG_MIGRATE) {
                /* successfully stole work from victim */
                local_data[local_data_len - 1] = recv_buffer[0];
                local_data_index = local_data_len - 1;
            } else if (tag == TAG_STOP_STEAL) {
                /* asked to stop stealing */
                stop_stealing_flag = 1;
            } else if (tag == TAG_TERMINATE) {
                /* received termination request. finish sending and exit loop */
                MPI_Wait(send_req, MPI_STATUSES_IGNORE);
                break;
            }
            MPI_Irecv(recv_buffer, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                      recv_req);
        }
        if (*send_req == MPI_REQUEST_NULL) {
            if (send_result_flag == 1) {
                /* */
                send_buffer[0] = nlocal_processed;
                MPI_Isend(send_buffer, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD, send_req);
                nlocal_processed = 0;
                send_result_flag = 0;
            } else if (stop_stealing_flag == 1) {
                /* tell master that this processes finishes stealing (no ongoing send) */
                MPI_Isend(NULL, 0, MPI_INT, 0, TAG_FINISH_STEAL, MPI_COMM_WORLD, send_req);
                stop_stealing_flag = 2;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
