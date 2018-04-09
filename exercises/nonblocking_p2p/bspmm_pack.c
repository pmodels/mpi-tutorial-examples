#include "bspmm.h"

int main(int argc, char **argv)
{
    int rank, nprocs;
    int mat_dim, blk_num;
    int work_id, work_id_len;
    double *mat_a, *mat_b, *mat_c;
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

        /* requests to receive prefetched blocks */
        MPI_Request recv_a_req = MPI_REQUEST_NULL, pf_recv_a_req = MPI_REQUEST_NULL;
        MPI_Request *pf_b_reqs, *recv_b_reqs, *pf_recv_b_reqs;

        pf_b_reqs = (MPI_Request *) malloc(sizeof(MPI_Request) * blk_num * 2);
        recv_b_reqs = pf_b_reqs;
        pf_recv_b_reqs = &pf_b_reqs[blk_num];

        /* buffers to keep prefetched blocks */
        double *pf_mem, *pf_local_a, *tmp_local_a, *original_local_a = local_a;
        double *local_bs, *pf_local_bs, *tmp_local_bs;

        MPI_Alloc_mem((2 * blk_num + 1) * BLK_DIM * BLK_DIM * sizeof(double), MPI_INFO_NULL,
                      &pf_mem);
        local_bs = pf_mem;
        pf_local_bs = local_bs + blk_num * BLK_DIM * BLK_DIM;
        pf_local_a = pf_local_bs + blk_num * BLK_DIM * BLK_DIM;

        int first_work_id = rank - 1;
        int i;

        /* prefetch blocks of A and B that are used in the first iteration */
        if (first_work_id < work_id_len) {
            MPI_Irecv(pf_local_a, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                      &pf_recv_a_req);
            for (i = 0; i < blk_num; i++)
                MPI_Irecv(&pf_local_bs[i * BLK_DIM * BLK_DIM], BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0,
                          MPI_COMM_WORLD, &pf_recv_b_reqs[i]);
        }

        for (work_id = first_work_id; work_id < work_id_len; work_id += nprocs - 1) {
            /* swap working local buffers and requests */
            recv_a_req = pf_recv_a_req;
            memcpy(recv_b_reqs, pf_recv_b_reqs, sizeof(MPI_Request) * blk_num);
            tmp_local_a = local_a;
            local_a = pf_local_a;
            pf_local_a = tmp_local_a;
            tmp_local_bs = local_bs;
            local_bs = pf_local_bs;
            pf_local_bs = tmp_local_bs;

            /* prefetch blocks of A and B that will be used in the next iteration */
            if (work_id + nprocs - 1 < work_id_len) {
                MPI_Irecv(pf_local_a, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                          &pf_recv_a_req);
                for (i = 0; i < blk_num; i++)
                    MPI_Irecv(&pf_local_bs[i * BLK_DIM * BLK_DIM], BLK_DIM * BLK_DIM, MPI_DOUBLE, 0,
                              0, MPI_COMM_WORLD, &pf_recv_b_reqs[i]);
            }

            /* wait prefetched blocks */
            MPI_Wait(&recv_a_req, MPI_STATUS_IGNORE);

            for (i = 0; i < blk_num; i++) {
                MPI_Wait(&recv_b_reqs[i], MPI_STATUS_IGNORE);

                /* compute Cij += Aik * Bkj only if both local_a and local_b are nonzero */
                if (is_zero_local(local_a) || is_zero_local(&local_bs[i * BLK_DIM * BLK_DIM])) {
                    memset(local_c, 0, BLK_DIM * BLK_DIM * sizeof(double));
                } else {
                    dgemm(local_a, &local_bs[i * BLK_DIM * BLK_DIM], local_c);
                }

                /* send C */
                MPI_Send(local_c, BLK_DIM * BLK_DIM, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }

        free(pf_b_reqs);
        MPI_Free_mem(pf_mem);
        local_a = original_local_a;
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
