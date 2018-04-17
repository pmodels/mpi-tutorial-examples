#include "bspmm.h"

/*
 * Block sparse matrix multiplication using shared memory and a global counter for workload
 * distribution.
 *
 * A, B, and C denote submatrices (BLK_DIM x BLK_DIM) and n is blk_num
 *
 * | C11 ... C1n |   | A11 ... A1n |    | B11 ... B1n |
 * |  . .     .  |   |  . .     .  |    |  . .     .  |
 * |  .  Cij  .  | = |  .  Aik  .  | *  |  .  Bkj  .  |
 * |  .     . .  |   |  .     . .  |    |  .     . .  |
 * | Cn1 ... Cnn |   | An1 ... Ann |    | Bn1 ... Bnn |
 *
 * This version of bspmm parallelizes i and j to eliminate atomic updates to C;
 * there are n^2 parallel computations of Cij += Aik * Bkj
 * Work id (0 <= id < n^2) is associated with each computation as follows
 *   (i, j) = (id / n, id % n)
 *
 * The master process allocates entire matrices A, B, and C. The worker processes
 * read submatrices of A and B directly from master's memory to calculate distinct
 * submatrices of C. Each worker stores the value of the submatrix of C that it
 * computes back into master's memory. The global counter is used for dynamic workload
 * distribution.
 */

int is_zero_local(double *local_mat);

void dgemm_increment_c(double *local_a, double *local_b, double *local_c);

void pack_global_to_local(double *local_mat, double *global_mat, int mat_dim, int global_i,
                          int global_j);

void unpack_local_to_global(double *global_mat, double *local_mat, int mat_dim, int global_i,
                            int global_j);

int main(int argc, char **argv)
{
    int rank, nprocs;
    int mat_dim, blk_num;
    int work_id, work_id_len;
    double *mat_a, *mat_b, *mat_c;
    double *local_a, *local_b, *local_c;
    double *mat_a_ptr, *mat_b_ptr, *mat_c_ptr;

    double *win_mem;
    int *counter_win_mem;
    MPI_Win win, win_counter;

    const int one = 1;

    double t1, t2;

    /* initialize MPI environment */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* create shared memory communicator */
    MPI_Comm shm_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);

    int shm_rank, shm_procs;
    MPI_Comm_size(shm_comm, &shm_procs);
    MPI_Comm_rank(shm_comm, &shm_rank);

    /* works only when all processes are in the same shared memory region */
    if (shm_procs != nprocs)
        MPI_Abort(MPI_COMM_WORLD, 1);

    /* argument checking and setting */
    if (setup(rank, nprocs, argc, argv, &mat_dim)) {
        MPI_Finalize();
        exit(0);
    }

    /* number of blocks in one dimension */
    blk_num = mat_dim / BLK_DIM;

    if (!rank) {
        /* create RMA windows */
        MPI_Win_allocate_shared(3 * mat_dim * mat_dim * sizeof(double), sizeof(double),
                                MPI_INFO_NULL, shm_comm, &win_mem, &win);
        MPI_Win_allocate_shared(sizeof(int), sizeof(int),
                                MPI_INFO_NULL, shm_comm, &counter_win_mem, &win_counter);
        mat_a = win_mem;
        mat_b = mat_a + mat_dim * mat_dim;
        mat_c = mat_b + mat_dim * mat_dim;

        /* initialize matrices and global counter */
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        init_mats(mat_dim, mat_a, mat_b, mat_c);
        MPI_Win_unlock(0, win); /* MEM_MODE: update to my private window becomes
                                 * visible in public window */

        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win_counter);
        *counter_win_mem = 0;
        MPI_Win_unlock(0, win_counter); /* MEM_MODE: update to my private window becomes
                                         * visible in public window */
    } else {
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, shm_comm, &win_mem, &win);
        MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, shm_comm, &counter_win_mem,
                                &win_counter);
    }

    /* acquire rank-0's pointer to all the three matrices */
    MPI_Aint win_sz;
    int disp_unit;
    MPI_Win_shared_query(win, 0, &win_sz, &disp_unit, &mat_a_ptr);
    mat_b_ptr = mat_a_ptr + mat_dim * mat_dim;
    mat_c_ptr = mat_b_ptr + mat_dim * mat_dim;

    /* allocate local buffer */
    local_a = (double *) malloc(3 * BLK_DIM * BLK_DIM * sizeof(double));
    local_b = local_a + BLK_DIM * BLK_DIM;
    local_c = local_b + BLK_DIM * BLK_DIM;

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win_counter);

    work_id_len = blk_num * blk_num;

    t1 = MPI_Wtime();

    do {
        /* read and increment global counter atomically */
        MPI_Fetch_and_op(&one, &work_id, MPI_INT, 0, 0, MPI_SUM, win_counter);
        MPI_Win_flush(0, win_counter);  /* MEM_MODE: update to target public window */
        if (work_id >= work_id_len)
            break;

        /* calculate global ids from the work_id */
        int global_i = work_id / blk_num;
        int global_j = work_id % blk_num;
        int global_k;

        /* initialize the value of local_c */
        memset(local_c, 0, BLK_DIM * BLK_DIM * sizeof(double));

        for (global_k = 0; global_k < blk_num; global_k++) {
            /* get block from mat_a in shared memory */
            pack_global_to_local(local_a, mat_a_ptr, mat_dim, global_i, global_k);

            if (is_zero_local(local_a))
                continue;

            /* get block from mat_b in shared memory */
            pack_global_to_local(local_b, mat_b_ptr, mat_dim, global_k, global_j);

            if (is_zero_local(local_b))
                continue;

            /* compute Cij += Aik * Bkj only if both local_a and local_b are nonzero */
            dgemm_increment_c(local_a, local_b, local_c);
        }

        /* store the value of local_c into the shared memory */
        unpack_local_to_global(mat_c_ptr, local_c, mat_dim, global_i, global_j);
    } while (work_id < work_id_len);

    /* sync here instead of right-after-store since each rank is updating distinct C-blocks and is not dependent on other C-blocks */
    MPI_Win_sync(win);  /* ensure completion of local updates before MPI_Barrier */
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();

    if (!rank) {
        MPI_Win_sync(win);      /* ensure remote updates are locally visible */
        check_mats(mat_a, mat_b, mat_c, mat_dim);
        printf("[%i] time: %f\n", rank, t2 - t1);
    }

    MPI_Win_unlock(0, win_counter);
    MPI_Win_unlock(0, win);

    free(local_a);
    MPI_Win_free(&win_counter);
    MPI_Win_free(&win);
    MPI_Comm_free(&shm_comm);

    MPI_Finalize();
    return 0;
}

int is_zero_local(double *local_mat)
{
    int i, j;

    for (i = 0; i < BLK_DIM; i++) {
        for (j = 0; j < BLK_DIM; j++) {
            if (local_mat[j + i * BLK_DIM] != 0.0)
                return 0;
        }
    }
    return 1;
}

void dgemm_increment_c(double *local_a, double *local_b, double *local_c)
{
    int i, j, k;

    for (j = 0; j < BLK_DIM; j++) {
        for (i = 0; i < BLK_DIM; i++) {
            for (k = 0; k < BLK_DIM; k++)
                local_c[j + i * BLK_DIM] += local_a[k + i * BLK_DIM] * local_b[j + k * BLK_DIM];
        }
    }
}

void pack_global_to_local(double *local_mat, double *global_mat, int mat_dim, int global_i,
                          int global_j)
{
    int i, j;
    int offset = global_i * BLK_DIM * mat_dim + global_j * BLK_DIM;

    for (i = 0; i < BLK_DIM; i++) {
        for (j = 0; j < BLK_DIM; j++)
            local_mat[j + i * BLK_DIM] = global_mat[offset + j + i * mat_dim];
    }
}

void unpack_local_to_global(double *global_mat, double *local_mat, int mat_dim, int global_i,
                            int global_j)
{
    int i, j;
    int offset = global_i * BLK_DIM * mat_dim + global_j * BLK_DIM;

    for (i = 0; i < BLK_DIM; i++) {
        for (j = 0; j < BLK_DIM; j++)
            global_mat[offset + j + i * mat_dim] = local_mat[j + i * BLK_DIM];
    }
}
