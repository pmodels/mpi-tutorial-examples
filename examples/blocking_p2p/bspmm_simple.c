#include "bspmm.h"

void add_local_to_global(double *global_mat, double *local_mat, int mat_dim, int global_i,
                         int global_j);
void copy_global_to_local(double *local_mat, double *global_mat, int mat_dim, int global_i,
                          int global_j);

int main(int argc, char **argv)
{
    int rank, nprocs;
    int mat_dim, blk_num;
    int work_id, work_id_len;
    double *mat_a = NULL, *mat_b = NULL, *mat_c = NULL, *mat_tmp_c = NULL;
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

    /* number of blocks in one dimension */
    blk_num = mat_dim / BLK_DIM;

    /* allocate memory for entire matrices */
    if (!rank) {
        /* initialize matrices */
        MPI_Alloc_mem(4 * mat_dim * mat_dim * sizeof(double), MPI_INFO_NULL, &mat_a);
        init_mats(mat_dim, mat_a, &mat_a, &mat_b, &mat_c);
        mat_tmp_c = mat_c + mat_dim * mat_dim;
    } else {
        MPI_Alloc_mem(3 * mat_dim * mat_dim * sizeof(double), MPI_INFO_NULL, &mat_a);
        mat_b = mat_a + mat_dim * mat_dim;
        mat_c = mat_b + mat_dim * mat_dim;
        memset(mat_c, 0, mat_dim * mat_dim * sizeof(double));
    }

    /* allocate local buffer */
    MPI_Alloc_mem(3 * BLK_DIM * BLK_DIM * sizeof(double), MPI_INFO_NULL, &local_a);
    local_b = local_a + BLK_DIM * BLK_DIM;
    local_c = local_b + BLK_DIM * BLK_DIM;

    MPI_Barrier(MPI_COMM_WORLD);

    t1 = MPI_Wtime();

    /* copy entire matrices to all the other processes */
    if (!rank) {
        /* send A and B to other processes */
        int i;
        for (i = 1; i < nprocs; i++) {
            MPI_Send(mat_a, mat_dim * mat_dim, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(mat_b, mat_dim * mat_dim, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    } else {
        /* receive A and B from master */
        MPI_Recv(mat_a, mat_dim * mat_dim, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(mat_b, mat_dim * mat_dim, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

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

    for (work_id = rank; work_id < work_id_len; work_id += nprocs) {
        int global_i = work_id / blk_num;
        int global_k = work_id % blk_num;
        int global_j;

        /* copy block from mat_a */
        copy_global_to_local(local_a, mat_a, mat_dim, global_i, global_k);
        if (is_zero(local_a))
            continue;

        for (global_j = 0; global_j < blk_num; global_j++) {
            /* copy block from mat_b */
            copy_global_to_local(local_b, mat_b, mat_dim, global_k, global_j);
            if (is_zero(local_b))
                continue;

            /* compute only if both local_a and local_b are nonzero */
            dgemm(local_a, local_b, local_c);

            /* write results to mat_c */
            add_local_to_global(mat_c, local_c, mat_dim, global_i, global_j);
        }
    }

    /* accumulate results */
    if (!rank) {
        int i, j, k;
        /* sum C computed by other processes */
        for (i = 1; i < nprocs; i++) {
            MPI_Recv(mat_tmp_c, mat_dim * mat_dim, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            for (j = 0; j < mat_dim; j++) {
                for (k = 0; k < mat_dim; k++)
                    mat_c[j + k * mat_dim] += mat_tmp_c[j + k * mat_dim];
            }
        }
    } else {
        /* send C to master */
        MPI_Send(mat_c, mat_dim * mat_dim, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
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

void add_local_to_global(double *global_mat, double *local_mat, int mat_dim, int global_i,
                         int global_j)
{
    int i, j;
    int offset = global_i * BLK_DIM * mat_dim + global_j * BLK_DIM;
    for (i = 0; i < BLK_DIM; i++) {
        for (j = 0; j < BLK_DIM; j++)
            global_mat[offset + j + i * mat_dim] += local_mat[j + i * BLK_DIM];
    }
}

void copy_global_to_local(double *local_mat, double *global_mat, int mat_dim, int global_i,
                          int global_j)
{
    int i, j;
    int offset = global_i * BLK_DIM * mat_dim + global_j * BLK_DIM;
    for (i = 0; i < BLK_DIM; i++) {
        for (j = 0; j < BLK_DIM; j++)
            local_mat[j + i * BLK_DIM] = global_mat[offset + j + i * mat_dim];
    }
}
