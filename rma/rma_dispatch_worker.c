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
    int next_idx = N_SUB_ARRAY - 1;
    double global_sum = 0.0;

    int data_array_len = 0;

    int comm_size, comm_rank;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    MPI_Win w_data, w_sub_array_len, w_sub_array_start;
    MPI_Win w_next_idx, w_global_sum;

    if (comm_rank == 0) {
        sub_array_len = malloc(N_SUB_ARRAY * sizeof(int));
        sub_array_start = malloc(N_SUB_ARRAY * sizeof(int));

        srand(0);

        for (int i = 0; i < N_SUB_ARRAY; i++) {
            sub_array_len[i] = rand() % SUB_ARRAY_LEN_VAR + SUB_ARRAY_LEN_BASE;
            sub_array_start[i] = data_array_len;
            data_array_len += sub_array_len[i];
        }

        data = malloc(data_array_len * sizeof(double));

        for (int i = 0; i < data_array_len; i++) {
            data[i] = (double) rand()/ (double) (RAND_MAX / 10000);
        }
        /* data init done */

        MPI_Win_create(data, data_array_len * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &w_data);
        MPI_Win_create(sub_array_len, N_SUB_ARRAY * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &w_sub_array_len);
        MPI_Win_create(sub_array_start, N_SUB_ARRAY * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &w_sub_array_start);
        MPI_Win_create(&next_idx, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &w_next_idx);
        MPI_Win_create(&global_sum, sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &w_global_sum);

        double correct_sum = 0.0;
        for (int i = 0; i < data_array_len; i++) {
            correct_sum += data[i];
        }

        /* wait completion */
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Win_free(&w_data);
        MPI_Win_free(&w_sub_array_len);
        MPI_Win_free(&w_sub_array_start);
        MPI_Win_free(&w_next_idx);
        MPI_Win_free(&w_global_sum);

        free(data);
        free(sub_array_len);
        free(sub_array_start);

        printf("The correct average of the array is %.4lf\n", correct_sum / data_array_len);
        printf("The average calculated by workers is %.4lf\n", global_sum / data_array_len);
    } else {
        /* worker */
        MPI_Win_create(NULL, 0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &w_data);
        MPI_Win_create(NULL, 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &w_sub_array_len);
        MPI_Win_create(NULL, 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &w_sub_array_start);
        MPI_Win_create(NULL, 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &w_next_idx);
        MPI_Win_create(NULL, 0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &w_global_sum);

        int my_work = 0;
        int sub_one = -1;
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, w_next_idx);
        MPI_Get_accumulate(&sub_one, 1, MPI_INT, &my_work, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_SUM, w_next_idx);
        MPI_Win_unlock(0, w_next_idx);

        while (my_work >= 0) {
            int work_start = 0;
            int work_len = 0;
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, w_sub_array_start);
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, w_sub_array_len);
            MPI_Get(&work_start, 1, MPI_INT, 0, my_work, 1, MPI_INT, w_sub_array_start);
            MPI_Get(&work_len, 1, MPI_INT, 0, my_work, 1, MPI_INT, w_sub_array_len);
            MPI_Win_unlock(0, w_sub_array_len);
            MPI_Win_unlock(0, w_sub_array_start);

            double *work_data = malloc(work_len * sizeof(double));
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, w_data);
            MPI_Get(work_data, work_len, MPI_DOUBLE, 0, work_start, work_len, MPI_DOUBLE, w_data);
            MPI_Win_unlock(0, w_data);

            double local_sum = 0.0;

            for (int i = 0; i < work_len; i++) {
                local_sum += work_data[i];
            }

            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, w_global_sum);
            MPI_Accumulate(&local_sum, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, MPI_SUM, w_global_sum);
            MPI_Win_unlock(0, w_global_sum);

            free(work_data);

            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, w_next_idx);
            MPI_Get_accumulate(&sub_one, 1, MPI_INT, &my_work, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_SUM, w_next_idx);
            MPI_Win_unlock(0, w_next_idx);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Win_free(&w_data);
        MPI_Win_free(&w_sub_array_len);
        MPI_Win_free(&w_sub_array_start);
        MPI_Win_free(&w_next_idx);
        MPI_Win_free(&w_global_sum);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
