/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * 2D stencil code using a blocking collective.
 *
 * 2D regular grid is divided into px * py blocks of grid points (px * py = # of processes.)
 * In every iteration, each process calls a blocking collective operation with derived data types to
 * exchange a halo with neighbors.
 */

#include "mpi.h"
#include "stencil_par.h"
#include "perf_stat.h"

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

    int rx, ry;
    int north, south, west, east;
    int bx, by, offx, offy;

    /* three heat sources */
    int sources[NSOURCES][2];
    int locnsources;            /* number of sources in my area */
    int locsources[NSOURCES][2];        /* sources local to my rank */

    int *send_counts, *recv_counts;
    int *sdispls, *rdispls;
    MPI_Datatype *types;

    int iter, i;

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

    /* determine my coordinates (x,y) -- rank=x*a+y in the 2d processor array */
    rx = rank % px;
    ry = rank / px;

    /* determine my four neighbors */
    north = (ry - 1) * px + rx;
    if (ry - 1 < 0)
        north = MPI_PROC_NULL;
    south = (ry + 1) * px + rx;
    if (ry + 1 >= py)
        south = MPI_PROC_NULL;
    west = ry * px + rx - 1;
    if (rx - 1 < 0)
        west = MPI_PROC_NULL;
    east = ry * px + rx + 1;
    if (rx + 1 >= px)
        east = MPI_PROC_NULL;

    /* decompose the domain */
    bx = n / px;        /* block size in x */
    by = n / py;        /* block size in y */
    offx = rx * bx;     /* offset in x */
    offy = ry * by;     /* offset in y */

    /* printf("%i (%i,%i) - w: %i, e: %i, n: %i, s: %i\n", rank, ry,rx,west,east,north,south); */

    /* initialize three heat sources */
    init_sources(bx, by, offx, offy, n, NSOURCES, sources, &locnsources, locsources);

    /* allocate working arrays */
    alloc_bufs(bx, by, &aold, &anew);

    /* create north-south datatype */
    MPI_Datatype north_south_type;
    MPI_Type_contiguous(bx, MPI_DOUBLE, &north_south_type);     /* Don't do this */
    MPI_Type_commit(&north_south_type);

    /* create east-west datatype */
    MPI_Datatype east_west_type;
    MPI_Type_vector(by, 1, bx + 2, MPI_DOUBLE, &east_west_type);
    MPI_Type_commit(&east_west_type);

    /* prepare arguments of alltoallw */
    send_counts = (int *) malloc(size * sizeof(int));
    memset(send_counts, 0, size * sizeof(int));

    recv_counts = (int *) malloc(size * sizeof(int));

    sdispls = (int *) malloc(size * sizeof(int));
    memset(sdispls, 0, size * sizeof(int));

    rdispls = (int *) malloc(size * sizeof(int));
    memset(rdispls, 0, size * sizeof(int));

    types = (MPI_Datatype *) malloc(size * sizeof(MPI_Datatype));
    for (i = 0; i < size; i++)
        types[i] = MPI_DATATYPE_NULL;

    if (north != MPI_PROC_NULL) {
        send_counts[north] = 1;
        sdispls[north] = ind(1, 1) * sizeof(double);
        rdispls[north] = ind(1, 0) * sizeof(double);
        types[north] = north_south_type;
    }
    if (south != MPI_PROC_NULL) {
        send_counts[south] = 1;
        sdispls[south] = ind(1, by) * sizeof(double);
        rdispls[south] = ind(1, by + 1) * sizeof(double);
        types[south] = north_south_type;
    }
    if (east != MPI_PROC_NULL) {
        send_counts[east] = 1;
        sdispls[east] = ind(bx, 1) * sizeof(double);
        rdispls[east] = ind(bx + 1, 1) * sizeof(double);
        types[east] = east_west_type;
    }
    if (west != MPI_PROC_NULL) {
        send_counts[west] = 1;
        sdispls[west] = ind(1, 1) * sizeof(double);
        rdispls[west] = ind(0, 1) * sizeof(double);
        types[west] = east_west_type;
    }

    /* use different count parameters because some MPI implementations do not consider displacements
     * in aliasing check */
    memcpy(recv_counts, send_counts, size * sizeof(int));

    PERF_TIMER_BEGIN(TIMER_EXEC);

    for (iter = 0; iter < niters; ++iter) {

        /* refresh heat sources */
        refresh_heat_source(bx, locnsources, locsources, energy, aold);

        PERF_TIMER_BEGIN(TIMER_COMM);
        MPI_Alltoallw(aold, send_counts, sdispls, types, aold, recv_counts, rdispls, types,
                      MPI_COMM_WORLD);
        PERF_TIMER_END(TIMER_COMM);

        /* update grid points */
        update_grid(bx, by, aold, anew, &heat);

        /* swap working arrays */
        tmp = anew;
        anew = aold;
        aold = tmp;

        /* optional - print image */
        if (iter == niters - 1)
            printarr_par(iter, anew, n, px, py, rx, ry, bx, by, offx, offy, ind_f, MPI_COMM_WORLD);
    }

    PERF_TIMER_END(TIMER_EXEC);

    MPI_Type_free(&east_west_type);
    MPI_Type_free(&north_south_type);

    /* free working arrays and communication buffers */
    free_bufs(aold, anew);

    /* get final heat in the system */
    MPI_Allreduce(&heat, &rheat, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("[%i] last heat: %f\n", rank, rheat);
        PERF_PRINT();
    }

    MPI_Finalize();
    return 0;
}
