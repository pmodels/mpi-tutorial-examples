/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * 2D stencil code using cartesian topology and a blocking neighborhood collective.
 *
 * 2D regular grid is divided into px * py blocks of grid points (px * py = # of processes.)
 * In every iteration, each process calls a blocking neighborhood collective with derived data types to
 * exchange grid points in a halo region with neighbors. Neighbors are calculated with cartesian topology.
 */

#include "stencil_par.h"

/* row-major order */
#define ind(i,j) ((j)*(bx+2)+(i))

int ind_f(int i, int j, int bx)
{
    return ind(i, j);
}

int main(int argc, char **argv)
{
    int rank, size;
    int n, energy, niters, px, py;

    int north, south, west, east;
    int bx, by, offx, offy;

    /* three heat sources */
    int sources[NSOURCES][2];
    int locnsources;            /* number of sources in my area */
    int locsources[NSOURCES][2];        /* sources local to my rank */

    double t1, t2;

    int iter;

    double *aold, *anew, *tmp;

    double heat, rheat;

    int final_flag;

    /* initialize MPI envrionment */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* argument checking and setting */
    setup(rank, size, argc, argv, &n, &energy, &niters, &px, &py, &final_flag);

    if (final_flag == 1) {
        MPI_Finalize();
        exit(0);
    }

    /* Create a communicator with a topology */
    MPI_Comm cart_comm;
    int dims[2], coords[2];
    int periods[2] = { 0, 0 };
    dims[0] = 0;
    dims[1] = 0;

    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    /* determine my four neighbors */
    MPI_Cart_shift(cart_comm, 0, 1, &west, &east);
    MPI_Cart_shift(cart_comm, 1, 1, &north, &south);

    /* decompose the domain */
    bx = n / px;        /* block size in x */
    by = n / py;        /* block size in y */
    offx = coords[0] * bx;      /* offset in x */
    offy = coords[1] * by;      /* offset in y */

    /* printf("%i (%i,%i) - w: %i, e: %i, n: %i, s: %i\n", rank, ry,rx,west,east,north,south); */

    /* allocate working arrays & communication buffers */
    aold = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */
    anew = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */

    memset(aold, 0, (bx + 2) * (by + 2) * sizeof(double));
    memset(anew, 0, (bx + 2) * (by + 2) * sizeof(double));

    /* initialize three heat sources */
    init_sources(bx, by, offx, offy, n, NSOURCES, sources, &locnsources, locsources);

    /* create east-west datatype */
    MPI_Datatype east_west_type;
    MPI_Type_vector(by, 1, bx + 2, MPI_DOUBLE, &east_west_type);
    MPI_Type_commit(&east_west_type);

    /* prepare arguments of neighborhood alltoallw (W, E, N, S) */
    int counts[4] = { 1, 1, bx, bx };
    MPI_Aint sdispls[4], rdispls[4];
    MPI_Datatype types[4] = { east_west_type, east_west_type, MPI_DOUBLE, MPI_DOUBLE };
    rdispls[0] = ind(0, 1) * sizeof(double);
    rdispls[1] = ind(bx + 1, 1) * sizeof(double);
    rdispls[2] = ind(1, 0) * sizeof(double);
    rdispls[3] = ind(1, by + 1) * sizeof(double);
    sdispls[0] = ind(1, 1) * sizeof(double);
    sdispls[1] = ind(bx, 1) * sizeof(double);
    sdispls[2] = ind(1, 1) * sizeof(double);
    sdispls[3] = ind(1, by) * sizeof(double);

    t1 = MPI_Wtime();   /* take time */

    for (iter = 0; iter < niters; ++iter) {

        /* refresh heat sources */
        refresh_heat_source(bx, locnsources, locsources, energy, aold);

        /* exchange data with neighbors */
        MPI_Neighbor_alltoallw(aold, counts, sdispls, types, aold, counts, rdispls, types,
                               cart_comm);

        /* update grid points */
        update_grid(bx, by, aold, anew, &heat);

        /* swap working arrays */
        tmp = anew;
        anew = aold;
        aold = tmp;

        /* optional - print image */
        if (iter == niters - 1)
            printarr_par(iter, anew, n, px, py, coords[0], coords[1],
                         bx, by, offx, offy, ind_f, MPI_COMM_WORLD);
    }

    t2 = MPI_Wtime();

    /* free working arrays and communication buffers */
    free(aold);
    free(anew);

    MPI_Type_free(&east_west_type);
    MPI_Comm_free(&cart_comm);

    /* get final heat in the system */
    MPI_Allreduce(&heat, &rheat, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
        printf("[%i] last heat: %f time: %f\n", rank, rheat, t2 - t1);

    MPI_Finalize();
    return 0;
}
