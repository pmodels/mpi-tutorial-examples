/*
 * 2D stencil code using shared memory
 *
 * Only run this code on a single node.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Input parameters */
int n;
int niters;

/* row-major order */
#define ind(i,j) ((j)*(bx+2)+(i))

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* create shared memory communicator */
    MPI_Comm shm_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);

    int shm_rank, shm_size;
    MPI_Comm_size(shm_comm, &shm_size);
    MPI_Comm_rank(shm_comm, &shm_rank);

    /* works only when all processes are in the same shared memory region */
    assert(shm_size == size);

    /* get input parameters from command line options */
    if (argc != 3) {
        printf("usage: %s <n> <niters>\n", argv[0]);
        return 0;
    }

    n = atoi(argv[1]);          /* n x n grid */
    niters = atoi(argv[2]);     /* number of iterations */

    int bx, by;
    bx = n;
    by = n;

    /* energy to be injected per iteration per source */
    int energy = 1.0;

    /* initialize three heat sources */
#define NSOURCES 3
    int sources[NSOURCES][2];
    sources[0][0] = n / 2;
    sources[0][1] = n / 2;
    sources[1][0] = n / 3;
    sources[1][1] = n / 3;
    sources[2][0] = n * 4 / 5;
    sources[2][1] = n * 8 / 9;

    /* allocate working arrays & communication buffers */
    MPI_Win win;
    double *win_mem;

    int grid_size = (bx + 2) * (by + 2);
    int mem_size = 2 * grid_size * sizeof(double); /* anew, aold */
    if (rank == 0) {
        MPI_Win_allocate_shared(mem_size, sizeof(double), MPI_INFO_NULL, shm_comm, &win_mem, &win);
    } else {
        void *dummy;
        MPI_Aint win_sz;
        int disp_unit;
        MPI_Win_allocate_shared(0, sizeof(double), MPI_INFO_NULL, shm_comm, &dummy, &win);
        MPI_Win_shared_query(win, 0, &win_sz, &disp_unit, &win_mem);
        assert(win_sz == mem_size);
        assert(disp_unit == sizeof(double));
        assert(win_mem);
    }

    double *aold, *anew;
    anew = win_mem;
    aold = win_mem + grid_size;

    /* devide the domain in y-dim */
    int j1, j2;
    int block = by / size;
    j1 = 1 + rank * block;
    if (rank == size - 1) {
        j2 = by + 1;
    } else {
        j2 = j1 + block;
    }

    /* determine which sources are in my patch */
    int locnsources = 0;         /* number of sources in my area */
    int locsources[NSOURCES][2]; /* sources local to my rank */

    for (int i = 0; i < NSOURCES; ++i) {
        /* NOTE: offset by 1 to account for halo zone */
        int locx = sources[i][0] + 1;
        int locy = sources[i][1] + 1;
        if (locx >= 1 && locx <= bx && locy >= j1 && locy < j2) {
            locsources[locnsources][0] = locx;
            locsources[locnsources][1] = locy;
            locnsources++;
        }
    }

    /* initialize working array
     * NOTE: include the halo zone for edge processes */
    int t_j1 = j1;
    int t_j2 = j2;
    if (rank == 0) {
        t_j1 = 0;
    } else if (rank == size - 1) {
        t_j2 = by + 2;
    }
    for (int j = t_j1; j < t_j2; ++j) {
        for (int i = 0; i < bx + 2; ++i) {
            aold[ind(i, j)] = 0.0;
            anew[ind(i, j)] = 0.0;
        }
    }

    double t_begin = MPI_Wtime();
    double last_heat;
    for (int iter = 0; iter < niters; ++iter) {
        /* refresh heat sources */
        for (int i = 0; i < locnsources; ++i) {
            aold[ind(locsources[i][0], locsources[i][1])] += energy;
        }

        MPI_Barrier(shm_comm);

        /* update grid points */
        double heat = 0.0;
        for (int i = 1; i < bx + 2; ++i) {
            for (int j = j1; j < j2; ++j) {
                anew[ind(i, j)] = aold[ind(i, j)] / 2.0 +
                                  (aold[ind(i - 1, j)] + aold[ind(i + 1, j)] +
                                   aold[ind(i, j - 1)] + aold[ind(i, j + 1)]) / 4.0 / 2.0;
                heat += anew[ind(i, j)];
            }
        }

        last_heat = heat;

        /* swap working arrays */
        double *tmp = anew;
        anew = aold;
        aold = tmp;

        /* Is this barrier necessary? */
        MPI_Barrier(shm_comm);
    }

    double t_end = MPI_Wtime();

    /* get final heat in the system */
    double rheat;
    MPI_Allreduce(&last_heat, &rheat, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("last heat: %f\n", rheat);
        printf("    Total run time: %.6f sec.\n", t_end - t_begin);
    }

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
