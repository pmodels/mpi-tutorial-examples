#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "mpi.h"

#define SUB_ARRAY_LEN_BASE 100
#define SUB_ARRAY_LEN_VAR 100
#define N_SUB_ARRAY 20

int main(int argc, char *argv[])
{
    double *data = NULL;
    int *sub_array_start = NULL;
    int *sub_array_len = NULL;
    double global_sum = 0.0;

    int data_array_len = 0;

    int comm_size, comm_rank;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    if (comm_rank == 0) {
        sub_array_len = malloc(N_SUB_ARRAY * sizeof(int));
        sub_array_start = malloc(N_SUB_ARRAY * sizeof(int));

        srand(0);

        /* data work list */
        for (int i = 0; i < N_SUB_ARRAY; i++) {
            sub_array_len[i] = rand() % SUB_ARRAY_LEN_VAR + SUB_ARRAY_LEN_BASE;
            sub_array_start[i] = data_array_len;
            data_array_len += sub_array_len[i];
        }

        /* init data */
        data = malloc(data_array_len * sizeof(double));

        for (int i = 0; i < data_array_len; i++) {
            data[i] = (double) rand()/ (double) (RAND_MAX / 10000);
        }

        /* calculate average for reference */
        double correct_sum = 0.0;

        /* wait completion */
        MPI_Barrier(MPI_COMM_WORLD);

        free(data);
        free(sub_array_len);
        free(sub_array_start);

        printf("The correct average of the array is %.4lf\n", correct_sum / data_array_len);
        printf("The average calculated by workers is %.4lf\n", global_sum / data_array_len);
    } else {
        /* worker */
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
