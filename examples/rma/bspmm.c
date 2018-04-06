/*
 * Copyright (c) 2014 Xin Zhao. All rights reserved.
 *
 * Author(s): Xin Zhao <xinzhao3@illinois.edu>
 *
 */

#include "bspmm.h"

int main(int argc, char **argv)
{
    int rank, nprocs;
    int mat_dim, blk_num;
    int work_id, work_id_len;
    double *mat_a = NULL, *mat_b = NULL, *mat_c = NULL;
    double *local_a, *local_b, *local_c;
    MPI_Aint disp_a, disp_b, disp_c;
    MPI_Aint offset_a, offset_b, offset_c;

    double *win_mem;
    MPI_Win win;

    double t1, t2;

    MPI_Datatype blk_dtp;

    /* initialize MPI environment */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* argument checking and setting */
    if (setup(rank, nprocs, argc, argv, &mat_dim)) {
        MPI_Finalize();
        exit(0);
    }

    /* number of blocks in one dimension */
    blk_num = mat_dim / BLK_DIM;

    if (!rank) {
        /* create RMA window */
        MPI_Win_allocate(3 * mat_dim * mat_dim * sizeof(double), sizeof(double),
                         MPI_INFO_NULL, MPI_COMM_WORLD, &win_mem, &win);

        /* initialize matrices */
        init_mats(mat_dim, win_mem, &mat_a, &mat_b, &mat_c);
    } else {
        MPI_Win_allocate(0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_mem, &win);
    }

    /* allocate local buffer */
    MPI_Alloc_mem(3 * BLK_DIM * BLK_DIM * sizeof(double), MPI_INFO_NULL, &local_a);
    local_b = local_a + BLK_DIM * BLK_DIM;
    local_c = local_b + BLK_DIM * BLK_DIM;

    /* create block datatype */
    MPI_Type_vector(BLK_DIM, BLK_DIM, mat_dim, MPI_DOUBLE, &blk_dtp);
    MPI_Type_commit(&blk_dtp);

    disp_a = 0;
    disp_b = disp_a + mat_dim * mat_dim;
    disp_c = disp_b + mat_dim * mat_dim;

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

    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);

    for (work_id = rank; work_id < work_id_len; work_id += nprocs) {
        int global_i = work_id / blk_num;
        int global_k = work_id % blk_num;
        int global_j;

        /* get block from mat_a */
        offset_a = global_i * BLK_DIM * mat_dim + global_k * BLK_DIM;
        MPI_Get(local_a, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, disp_a + offset_a, 1, blk_dtp, win);
        MPI_Win_flush(0, win);

        if (is_zero_local(local_a))
            continue;

        for (global_j = 0; global_j < blk_num; global_j++) {
            /* get block from mat_b */
            offset_b = global_k * BLK_DIM * mat_dim + global_j * BLK_DIM;
            MPI_Get(local_b, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, disp_b + offset_b, 1, blk_dtp, win);
            MPI_Win_flush(0, win);

            if (is_zero_local(local_b))
                continue;

            /* compute only if both local_a and local_b are nonzero */
            dgemm(local_a, local_b, local_c);

            /* accumulate block to mat_c */
            offset_c = global_i * BLK_DIM * mat_dim + global_j * BLK_DIM;
            MPI_Accumulate(local_c, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, disp_c + offset_c, 1, blk_dtp,
                           MPI_SUM, win);
            MPI_Win_flush(0, win);
        }
    }

    MPI_Win_unlock(0, win);

    t2 = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    if (!rank) {
        check_mats(mat_a, mat_b, mat_c, mat_dim);
        printf("[%i] time: %f\n", rank, t2 - t1);
    }

    MPI_Type_free(&blk_dtp);
    MPI_Free_mem(local_a);
    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
