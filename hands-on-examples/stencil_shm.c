/*
 * 2D stencil code using nonblocking send/receive with derived data types.
 *
 * 2D regular grid are divided into px * py blocks of grid points (px * py = # of processes.)
 * In every iteration, each process calls nonblocking operations with derived data types to exchange
 * grid points in a halo region with its neighbors.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Input parameters */
int n, niters, px, py;

/* row-major order */
#define ind(i,j) ((j)*(n+2)+(i))

int main(int argc, char **argv)
{
    /* initialize MPI envrionment */
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* get input parameters from command line options */
    if (argc != 5) {
        if (rank == 0)
            printf("usage: %s <n> <niters> <px> <py>\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    n = atoi(argv[1]);      /* nxn grid */
    niters = atoi(argv[2]); /* number of iterations */
    px = atoi(argv[3]);     /* 1st dim processes */
    py = atoi(argv[4]);     /* 2nd dim processes */
    assert(px * py == size);
    assert(n % px == 0);
    assert(n % py == 0);

    /* determine my coordinates (x,y) -- rank=x*a+y in the 2d processor array */
    int rx, ry;
    rx = rank % px;
    ry = rank / px;

    /* determine my four neighbors */
    int north, south, west, east;
    north = (ry > 0)      ? (ry - 1) * px + rx : MPI_PROC_NULL;
    south = (ry < py - 1) ? (ry + 1) * px + rx : MPI_PROC_NULL;
    west =  (rx > 0)      ? ry * px + rx - 1   : MPI_PROC_NULL;
    east =  (rx < px - 1) ? ry * px + rx + 1   : MPI_PROC_NULL;

    /* decompose the domain */
    int bx, by;
    bx = n / px;        /* block size in x */
    by = n / py;        /* block size in y */

    /* find my portion of the domain */
    int start_x, start_y, end_x, end_y;
    start_x = bx * rx + 1;
    start_y = by * ry + 1;
    end_x = start_x + bx;
    end_y = start_y + by;

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

    /* determine which sources are in my patch */
    int locnsources = 0;         /* number of sources in my area */
    int locsources[NSOURCES][2]; /* sources local to my rank */

    for (int i = 0; i < NSOURCES; ++i) {
        /* NOTE: offset by 1 to account for halo zone */
        int locx = sources[i][0] + 1;
        int locy = sources[i][1] + 1;
        if (locx >= start_x && locx < end_x && locy >= start_y && locy < end_y) {
            locsources[locnsources][0] = locx;
            locsources[locnsources][1] = locy;
            locnsources++;
        }
    }

    /* create shared memory communicator */
    MPI_Comm shm_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);

    int shm_rank, shm_size;
    MPI_Comm_size(shm_comm, &shm_size);
    MPI_Comm_rank(shm_comm, &shm_rank);

    /* works only when all processes are in the same shared memory region */
    assert(shm_size == size);

    /* allocate working arrays & communication buffers */
    MPI_Win win;
    double *win_mem;

    int grid_size = (n + 2) * (n + 2);
    int mem_size = 2 * grid_size * sizeof(double); /* anew, aold */
    if (rank == 0) {
        MPI_Win_allocate_shared(mem_size, sizeof(double), MPI_INFO_NULL, shm_comm, &win_mem, &win);
        memset(win_mem, 0, mem_size);
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

    MPI_Barrier(shm_comm);

    double *aold, *anew;
    anew = win_mem;
    aold = win_mem + grid_size;

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
        for (int i = start_x; i < end_x; ++i) {
            for (int j = start_y; j < end_y; ++j) {
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
    }

    double t_end = MPI_Wtime();

    /* get final heat in the system */
    double rheat;
    MPI_Allreduce(&last_heat, &rheat, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("last heat: %f\n", rheat);
        printf("    Total computation time: %.6f sec.\n", t_end - t_begin);
    }

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
