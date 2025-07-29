/*
 * 2D stencil code using nonblocking send/receive with derived data types.
 *
 * 2D regular grid are divided into px * py blocks of grid points (px * py = # of processes.)
 * In every iteration, each process calls nonblocking operations with derived data types to exchange
 * grid points in a halo region with its neighbors.
 */

#include <omp.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Input parameters */
int n, niters, px, py;

/* row-major order */
#define ind(i,j) ((j)*(bx+2)+(i))

int main(int argc, char **argv)
{
    /* initialize MPI envrionment */
    int thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);
    assert(thread_level >= MPI_THREAD_MULTIPLE);

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

    /* decompose the domain */
    int bx, by;
    bx = n / px;        /* block size in x */
    by = n / py;        /* block size in y */

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

    /* allocate working arrays & communication buffers.
     * NOTE: Include 1-wide halo zones on each side.
     * NOTE: do not initialize here */
    double *aold_shared, *anew_shared;
    anew_shared = malloc((bx + 2) * (by + 2) * sizeof(double));
    aold_shared = malloc((bx + 2) * (by + 2) * sizeof(double));

    double t_begin = MPI_Wtime();
    double last_heat = 0.0;
#pragma omp parallel num_threads(4)
  {
    int nthreads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();

    /* divide blocks in x amongst threads */
    int Thx = bx / nthreads;
    assert(Thx > 0);

    int xstart = thread_id * Thx + 1;
    int xend = (thread_id == nthreads - 1) ? bx + 1 : xstart + Thx;
    int ystart = 1;
    int yend = by + 1;

    /* determine which sources are in my patch */
    int locnsources = 0;         /* number of sources in my area */
    int locsources[NSOURCES][2]; /* sources local to my rank */
    for (int i = 0; i < NSOURCES; ++i) {
        /* NOTE: offset by 1 to account for halo zone */
        int locx = sources[i][0] - rx * bx + 1;
        int locy = sources[i][1] - ry * by + 1;
        if (locx >= xstart && locx < xend && locy >= ystart && locy < yend) {
            locsources[locnsources][0] = locx;
            locsources[locnsources][1] = locy;
            locnsources++;
        }
    }

    /* Initialize aold and anew using "first touch", *including boundaries* */
    double *anew = anew_shared;
    double *aold = aold_shared;

    int x1 = (xstart == 1) ? 0 : xstart;
    int x2 = (xend == bx + 1) ? bx + 2 : xend;
    int y1 = (ystart == 1) ? 0 : ystart;
    int y2 = (yend == by + 1) ? by + 2 : yend;
    for (int j = y1; j < y2; ++j) {
        for (int i = x1; i < x2; ++i) {
            aold[ind(i, j)] = 0.0;
            anew[ind(i, j)] = 0.0;
        }
    }

    /* determine my four neighbors */
    /* NOTE: only the thread at the edge need to communicate */
    int north, south, west, east;
    north = (ystart == 1    && ry > 0)      ? (ry - 1) * px + rx : MPI_PROC_NULL;
    south = (yend == by + 1 && ry < py - 1) ? (ry + 1) * px + rx : MPI_PROC_NULL;
    west =  (xstart == 1    && rx > 0)      ? ry * px + rx - 1   : MPI_PROC_NULL;
    east =  (xend == bx + 1 && rx < px - 1) ? ry * px + rx + 1   : MPI_PROC_NULL;

    /* create north-south datatype */
    MPI_Datatype north_south_type;
    MPI_Type_contiguous(xend - xstart, MPI_DOUBLE, &north_south_type);
    MPI_Type_commit(&north_south_type);

    /* create east-west datatype */
    MPI_Datatype east_west_type;
    MPI_Type_vector(yend - ystart, 1, bx + 2, MPI_DOUBLE, &east_west_type);
    MPI_Type_commit(&east_west_type);

    double thread_last_heat;
    for (int iter = 0; iter < niters; ++iter) {
        /* refresh heat sources */
        for (int i = 0; i < locnsources; ++i) {
            aold[ind(locsources[i][0], locsources[i][1])] += energy;
        }

#pragma omp barrier

        /* exchange data with neighbors */
        /* use sender's thread_id as tag.
         * NOTE: north and south neighbors share the same thread id,
         *       east and west neighbors do not share the same thread_id */
        int tag = thread_id;
        int east_tag = 0;
        int west_tag = nthreads - 1;

        MPI_Request reqs[8];
        MPI_Isend(&aold[ind(xstart, ystart)],     1, north_south_type, north, tag,      MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(&aold[ind(xstart, yend - 1)],   1, north_south_type, south, tag,      MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(&aold[ind(xend - 1, ystart)],   1, east_west_type,    east, tag,      MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(&aold[ind(xstart, ystart)],     1, east_west_type,    west, tag,      MPI_COMM_WORLD, &reqs[3]);
        MPI_Irecv(&aold[ind(xstart, ystart - 1)], 1, north_south_type, north, tag,      MPI_COMM_WORLD, &reqs[4]);
        MPI_Irecv(&aold[ind(xstart, yend)],       1, north_south_type, south, tag,      MPI_COMM_WORLD, &reqs[5]);
        MPI_Irecv(&aold[ind(xend, ystart)],       1, east_west_type,    east, east_tag, MPI_COMM_WORLD, &reqs[6]);
        MPI_Irecv(&aold[ind(xstart - 1, ystart)], 1, east_west_type,    west, west_tag, MPI_COMM_WORLD, &reqs[7]);
        MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);

        /* update grid points */
        double heat = 0.0;
        for (int i = xstart; i < xend; ++i) {
            for (int j = ystart; j < yend; ++j) {
                anew[ind(i, j)] = aold[ind(i, j)] / 2.0 +
                                  (aold[ind(i - 1, j)] + aold[ind(i + 1, j)] +
                                   aold[ind(i, j - 1)] + aold[ind(i, j + 1)]) / 4.0 / 2.0;
                heat += anew[ind(i, j)];
            }
        }

        thread_last_heat = heat;

        /* swap working arrays */
        double *tmp = anew;
        anew = aold;
        aold = tmp;

    } /* for - iter */

    MPI_Type_free(&east_west_type);
    MPI_Type_free(&north_south_type);

#pragma omp critical
    {
        last_heat += thread_last_heat;
    }
  } /* omp parallel */

    double t_end = MPI_Wtime();

    /* free working arrays and communication buffers */
    free(aold_shared);
    free(anew_shared);

    /* get final heat in the system */
    double rheat;
    MPI_Allreduce(&last_heat, &rheat, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("last heat: %f\n", rheat);
        printf("    Total computation time: %.6f sec.\n", t_end - t_begin);
    }

    MPI_Finalize();
    return 0;
}
