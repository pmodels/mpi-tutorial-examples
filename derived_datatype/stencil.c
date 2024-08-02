/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * 2D stencil code using nonblocking send/receive with derived data types.
 *
 * 2D regular grid are divided into px * py blocks of grid points (px * py = # of processes.)
 * In every iteration, each process calls nonblocking operations with derived data types to exchange
 * grid points in a halo region with its neighbors.
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

    PERF_TIMER_BEGIN(TIMER_EXEC);

    for (iter = 0; iter < niters; ++iter) {

        /* refresh heat sources */
        refresh_heat_source(bx, locnsources, locsources, energy, aold);

        /* exchange data with neighbors */
        MPI_Request reqs[8];
        PERF_TIMER_BEGIN(TIMER_COMM);
        MPI_Isend(&aold[ind(1, 1)] /* north */ , 1, north_south_type, north, 9, MPI_COMM_WORLD,
                  &reqs[0]);
        MPI_Isend(&aold[ind(1, by)] /* south */ , 1, north_south_type, south, 9, MPI_COMM_WORLD,
                  &reqs[1]);
        MPI_Isend(&aold[ind(bx, 1)] /* east */ , 1, east_west_type, east, 9, MPI_COMM_WORLD,
                  &reqs[2]);
        MPI_Isend(&aold[ind(1, 1)] /* west */ , 1, east_west_type, west, 9, MPI_COMM_WORLD,
                  &reqs[3]);
        MPI_Irecv(&aold[ind(1, 0)] /* north */ , 1, north_south_type, north, 9, MPI_COMM_WORLD,
                  &reqs[4]);
        MPI_Irecv(&aold[ind(1, by + 1)] /* south */ , 1, north_south_type, south, 9, MPI_COMM_WORLD,
                  &reqs[5]);
        MPI_Irecv(&aold[ind(bx + 1, 1)] /* east */ , 1, east_west_type, east, 9, MPI_COMM_WORLD,
                  &reqs[6]);
        MPI_Irecv(&aold[ind(0, 1)] /* west */ , 1, east_west_type, west, 9, MPI_COMM_WORLD,
                  &reqs[7]);
        MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
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
