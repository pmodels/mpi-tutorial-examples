#include "bspmm.h"

int main(int argc, char **argv)
{
    int rank, nprocs, count;
    int mat_dim, blk_num;
    int work_id, work_id_len;
    double *mat_a = NULL, *mat_b = NULL, *mat_c = NULL;
    double *local_a, *local_b, *local_c;
    MPI_Status status;

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
        mat_b = mat_a + mat_dim * mat_dim;
        mat_c = mat_b + mat_dim * mat_dim;
        init_mats(mat_dim, mat_a, mat_b, mat_c);
    } else {
        MPI_Alloc_mem(3 * mat_dim * mat_dim * sizeof(double), MPI_INFO_NULL, &mat_a);
        mat_b = mat_a + mat_dim * mat_dim;
        mat_c = mat_b + mat_dim * mat_dim;
        memset(mat_a, 0, 3 * mat_dim * mat_dim * sizeof(double));
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

    /* distribute blocks */
    if (!rank) {
        /* send A and B to workers */
        for (work_id = 0; work_id < work_id_len; work_id++) {
            int target = work_id % (nprocs - 1) + 1;
            int global_i = work_id / blk_num;
            int global_k = work_id % blk_num;
            int global_j;

            /* send A */
            if (is_zero_global(mat_a, mat_dim, global_i, global_k)) {
                /* send empty message */
                MPI_Send(NULL, 0, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
            } else {
                pack_global_to_local(local_a, mat_a, mat_dim, global_i, global_k);
                MPI_Send(local_a, BLK_DIM * BLK_DIM, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
            }

            /* send B */
            for (global_j = 0; global_j < blk_num; global_j++) {
                if (is_zero_global(mat_b, mat_dim, global_k, global_j)) {
                    /* send empty message */
                    MPI_Send(NULL, 0, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
                } else {
                    pack_global_to_local(local_b, mat_b, mat_dim, global_k, global_j);
                    MPI_Send(local_b, BLK_DIM * BLK_DIM, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
                }
            }
        }
    } else {
        /* receive A and B from master */
        for (work_id = rank - 1; work_id < work_id_len; work_id += nprocs - 1) {
            int global_i = work_id / blk_num;
            int global_k = work_id % blk_num;
            int global_j;

            /* receive A */
            MPI_Recv(local_a, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &count);
            if (count)
                unpack_local_to_global(mat_a, local_a, mat_dim, global_i, global_k);

            /* receive B */
            for (global_j = 0; global_j < blk_num; global_j++) {
                MPI_Recv(local_b, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Get_count(&status, MPI_DOUBLE, &count);
                if (count)
                    unpack_local_to_global(mat_b, local_b, mat_dim, global_k, global_j);
            }
        }
    }

    /* compute Cij += Aik * Bkj */
    if (rank) {
        for (work_id = rank - 1; work_id < work_id_len; work_id += nprocs - 1) {
            int global_i = work_id / blk_num;
            int global_k = work_id % blk_num;
            int global_j;

            /* copy block from mat_a */
            pack_global_to_local(local_a, mat_a, mat_dim, global_i, global_k);
            if (is_zero_local(local_a))
                continue;

            for (global_j = 0; global_j < blk_num; global_j++) {
                /* copy block from mat_b */
                pack_global_to_local(local_b, mat_b, mat_dim, global_k, global_j);
                if (is_zero_local(local_b))
                    continue;

                /* compute only if both local_a and local_b are nonzero */
                dgemm(local_a, local_b, local_c);

                /* write results to mat_c */
                add_local_to_global(mat_c, local_c, mat_dim, global_i, global_j);
            }
        }
    }

    /* accumulate results */
    if (!rank) {
        /* sum C computed by workers */
        char *recv_blks_c = (char *) calloc(nprocs * blk_num, sizeof(char));

        for (work_id = 0; work_id < work_id_len; work_id++) {
            int target = work_id % (nprocs - 1) + 1;
            int global_i = work_id / blk_num;
            int global_j;

            if (recv_blks_c[global_i * nprocs + target] == 1)
                continue;       /* Cij has been already added */

            for (global_j = 0; global_j < blk_num; global_j++) {
                /* receive C from target */
                MPI_Recv(local_c, BLK_DIM * BLK_DIM, MPI_DOUBLE, target, 0, MPI_COMM_WORLD,
                         &status);
                MPI_Get_count(&status, MPI_DOUBLE, &count);
                if (count)
                    add_local_to_global(mat_c, local_c, mat_dim, global_i, global_j);
            }
            recv_blks_c[global_i * nprocs + target] = 1;
        }
        free(recv_blks_c);
    } else {
        /* send C to master */
        char *send_blks_c = (char *) calloc(blk_num, sizeof(char));

        for (work_id = rank - 1; work_id < work_id_len; work_id += nprocs - 1) {
            int global_i = work_id / blk_num;
            int global_j;

            if (send_blks_c[global_i] == 1)
                continue;       /* Cij has been already sent */

            for (global_j = 0; global_j < blk_num; global_j++) {
                /* send C to master */
                if (is_zero_global(mat_c, mat_dim, global_i, global_j)) {
                    MPI_Send(NULL, 0, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                } else {
                    pack_global_to_local(local_c, mat_c, mat_dim, global_i, global_j);
                    MPI_Send(local_c, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
            }
            send_blks_c[global_i] = 1;
        }
        free(send_blks_c);
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
