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
#define ind(i,j) ((j)*(bx+2)+(i))

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

    // printf("%i (%i,%i) - w: %i, e: %i, n: %i, s: %i\n", rank, ry,rx,west,east,north,south);

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
        int locx = sources[i][0] - rx * bx + 1;
        int locy = sources[i][1] - ry * by + 1;
        if (locx >= 1 && locx <= bx && locy >= 1 && locy <= by) {
            locsources[locnsources][0] = locx;
            locsources[locnsources][1] = locy;
            locnsources++;
        }
    }

    /* allocate working arrays & communication buffers.
     * NOTE: Include 1-wide halo zones on each side. */
    double *aold, *anew;
    anew = malloc((bx + 2) * (by + 2) * sizeof(double));
    aold = malloc((bx + 2) * (by + 2) * sizeof(double));
    /* initialize */
    memset(aold, 0, (bx + 2) * (by + 2) * sizeof(double));
    memset(anew, 0, (bx + 2) * (by + 2) * sizeof(double));

    /* allocate communication buffers. */
    /* NOTE: since we don't need to pack north and south data, we don't need malloc them */
    double *sbufnorth, *sbufsouth, *sbufeast, *sbufwest; /* send buffers */
    double *rbufnorth, *rbufsouth, *rbufeast, *rbufwest; /* recv buffers */
    sbufeast = (double *) malloc(by * sizeof(double));
    sbufwest = (double *) malloc(by * sizeof(double));
    rbufeast = (double *) malloc(by * sizeof(double));
    rbufwest = (double *) malloc(by * sizeof(double));

    double t_begin = MPI_Wtime();
    double last_heat;
    for (int iter = 0; iter < niters; ++iter) {
        /* refresh heat sources */
        for (int i = 0; i < locnsources; ++i) {
            aold[ind(locsources[i][0], locsources[i][1])] += energy;
        }

        /* pack data for send */
        sbufnorth = &aold[ind(1, 1)];
        sbufsouth = &aold[ind(1, by)];
        for (int j = 1; j < by + 1; j++) {
            sbufeast[j - 1] = aold[ind(bx, j)];
            sbufwest[j - 1] = aold[ind(1, j)];
        }

        /* exchange data with neighbors */
        MPI_Request reqs[8];
        MPI_Isend(sbufnorth, bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(sbufsouth, bx, MPI_DOUBLE, south, 9, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(sbufeast,  by, MPI_DOUBLE,  east, 9, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(sbufwest,  by, MPI_DOUBLE,  west, 9, MPI_COMM_WORLD, &reqs[3]);

        rbufnorth = &aold[ind(1, 0)];
        rbufsouth = &aold[ind(1, by + 1)];
        MPI_Irecv(rbufnorth, bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD, &reqs[4]);
        MPI_Irecv(rbufsouth, bx, MPI_DOUBLE, south, 9, MPI_COMM_WORLD, &reqs[5]);
        MPI_Irecv(rbufeast,  by, MPI_DOUBLE,  east, 9, MPI_COMM_WORLD, &reqs[6]);
        MPI_Irecv(rbufwest,  by, MPI_DOUBLE,  west, 9, MPI_COMM_WORLD, &reqs[7]);
        MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);

        /* unpack data after receive */
        if (east != MPI_PROC_NULL) {
            for (int j = 1; j < by + 1; j++) {
                aold[ind(bx + 1, j)] = rbufeast[j - 1];
            }
        }
        if (west != MPI_PROC_NULL) {
            for (int j = 1; j < by + 1; j++) {
                aold[ind(0, j)] = rbufwest[j - 1];
            }
        }

        /* update grid points */
        double heat = 0.0;
        for (int i = 1; i < bx + 1; ++i) {
            for (int j = 1; j < by + 1; ++j) {
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

    /* free working arrays and communication buffers */
    free(aold);
    free(anew);
    free(sbufeast);
    free(sbufwest);
    free(rbufeast);
    free(rbufwest);

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
