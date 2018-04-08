#include "bspmm.h"

int main(int argc, char **argv)
{
    int rank, nprocs;
    int mat_dim, blk_num;
    int work_id, work_id_len;
    double *mat_a = NULL, *mat_b = NULL, *mat_c = NULL;
    double *local_a, *local_b, *local_c;

    double t1, t2;

    /* initialize MPI environment */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* argument checking and setting */
    if (setup(rank, nprocs, argc, argv, &mat_dim)) {
        MPI_Finalize();
        exit(0);
    }
    if (nprocs == 1) {
        printf("nprocs must be more than 1.\n");
        MPI_Finalize();
        exit(0);
    }

    /* number of blocks in one dimension */
    blk_num = mat_dim / BLK_DIM;

    /* initialize matrices */
    if (!rank) {
        MPI_Alloc_mem(3 * mat_dim * mat_dim * sizeof(double), MPI_INFO_NULL, &mat_a);
        init_mats(mat_dim, mat_a, &mat_a, &mat_b, &mat_c);
    }

    /* allocate local buffer */
    MPI_Alloc_mem(3 * BLK_DIM * BLK_DIM * sizeof(double), MPI_INFO_NULL, &local_a);
    local_b = local_a + BLK_DIM * BLK_DIM;
    local_c = local_b + BLK_DIM * BLK_DIM;

    MPI_Barrier(MPI_COMM_WORLD);

    t1 = MPI_Wtime();

    /*
     * A, B, and C denote submatrices (BLK_DIM x BLK_DIM) and n is blk_num
     *
     * | C11 ... C1n |   | A11 ... A1n |    | B11 ... B1n |
     * |  . .     .  |   |  . .     .  |    |  . .     .  |
     * |  .  Cij  .  | = |  .  Aik  .  | *  |  .  Bkj  .  |
     * |  .     . .  |   |  .     . .  |    |  .     . .  |
     * | Cn1 ... Cnn |   | An1 ... Ann |    | Bn1 ... Bnn |
     *
     * bspmm parallelizes i and k; there are n^2 parallel computations of Cij += Aik * Bkj
     * Work id (0 <= id < n^2) is associated with each computation as follows
     *   (i, k) = (id / n, id % n)
     * Note Cij must be updated atomically
     */

    work_id_len = blk_num * blk_num;

    if (!rank) {
        /* distribute A and B and receive results */
        int iter, niters = (work_id_len + nprocs - 2) / (nprocs - 1);

        for (iter = 0; iter < niters; iter++) {
            int target;
            /* not all workers work in the last iteration */
            int target_end = iter != niters - 1 ? nprocs : work_id_len - iter * (nprocs - 1) + 1;
            int global_j;

            /* send A to workers */
            for (target = 1; target < target_end; target++) {
                int work_id = iter * (nprocs - 1) + target - 1;
                int global_i = work_id / blk_num;
                int global_k = work_id % blk_num;

                pack_global_to_local(local_a, mat_a, mat_dim, global_i, global_k);
                MPI_Send(local_a, BLK_DIM * BLK_DIM, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
            }

            /* send B to workers and receive C one by one */
            for (global_j = 0; global_j < blk_num; global_j++) {
                /* send B */
                for (target = 1; target < target_end; target++) {
                    int work_id = iter * (nprocs - 1) + target - 1;
                    int global_k = work_id % blk_num;

                    pack_global_to_local(local_b, mat_b, mat_dim, global_k, global_j);
                    MPI_Send(local_b, BLK_DIM * BLK_DIM, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
                }

                /* receive C */
                for (target = 1; target < target_end; target++) {
                    int work_id = iter * (nprocs - 1) + target - 1;
                    int global_i = work_id / blk_num;

                    MPI_Recv(local_c, BLK_DIM * BLK_DIM, MPI_DOUBLE, target, 0, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    add_local_to_global(mat_c, local_c, mat_dim, global_i, global_j);
                }
            }
        }
    } else {
        /* receive A and B from master and return result */
        for (work_id = rank - 1; work_id < work_id_len; work_id += nprocs - 1) {
            int i;

            /* receive A */
            MPI_Recv(local_a, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

            /* receive B */
            for (i = 0; i < blk_num; i++) {
                MPI_Recv(local_b, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                /* compute Cij += Aik * Bkj */
                if (is_zero_local(local_a) || is_zero_local(local_b)) {
                    memset(local_c, 0, BLK_DIM * BLK_DIM * sizeof(double));
                } else {
                    /* compute only if both local_a and local_b are nonzero */
                    dgemm(local_a, local_b, local_c);
                }

                /* send C */
                MPI_Send(local_c, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    t2 = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    if (!rank) {
        check_mats(mat_a, mat_b, mat_c, mat_dim);
        printf("[%i] time: %f\n", rank, t2 - t1);
    }

    MPI_Free_mem(local_a);
    MPI_Free_mem(mat_a);
    MPI_Finalize();
    return 0;
}
