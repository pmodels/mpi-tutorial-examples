/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * 2D stencil code using RMA put operations and PSCW synchronization.
 *
 * 2D regular grid is divided into px * py blocks of grid points (px * py = # of processes.)
 * In every iteration, each process starts an RMA epoch (post-start) with neighbors and issues RMA
 * operations to put its outer grid points to neighbors' halo regions. RMA operations are synchronized
 * by closing the epoch (complete-wait).
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

    int rx, ry;
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

    int grid_size;              /* grid size */
    double *win_mem;            /* window memory */
    MPI_Win win;                /* RMA window */


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

    /* allocate RMA window */
    grid_size = (bx + 2) * (by + 2);    /* process-local grid (including halos (thus +2)) */

    /* create RMA window upon working array */
    MPI_Win_allocate(2 * grid_size * sizeof(double), sizeof(double), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &win_mem, &win);
    memset(win_mem, 0, 2 * grid_size * sizeof(double));

    anew = win_mem;
    aold = win_mem + grid_size; /* second half is aold! */

    /* initialize three heat sources */
    init_sources(bx, by, offx, offy, n, NSOURCES, sources, &locnsources, locsources);

    /* create east-west datatype */
    MPI_Datatype east_west_type;
    MPI_Type_vector(by, 1, bx + 2, MPI_DOUBLE, &east_west_type);
    MPI_Type_commit(&east_west_type);

    /* create neighbors group */
    MPI_Group win_group, neighbors_group;
    int neighbors[4], num_neighbors = 0;
    if (north != MPI_PROC_NULL)
        neighbors[num_neighbors++] = north;
    if (south != MPI_PROC_NULL)
        neighbors[num_neighbors++] = south;
    if (east != MPI_PROC_NULL)
        neighbors[num_neighbors++] = east;
    if (west != MPI_PROC_NULL)
        neighbors[num_neighbors++] = west;

    MPI_Win_get_group(win, &win_group);
    MPI_Group_incl(win_group, num_neighbors, neighbors, &neighbors_group);
    MPI_Group_free(&win_group);

    t1 = MPI_Wtime();   /* take time */

    for (iter = 0; iter < niters; ++iter) {

        int offset;

        /* refresh heat sources */
        refresh_heat_source(bx, locnsources, locsources, energy, aold);

        /* exchange data with neighbors */
        MPI_Win_post(neighbors_group, 0, win);  /* MEM_MODE: update to my private window
                                                 * becomes visible in public window */
        MPI_Win_start(neighbors_group, 0, win);

        offset = grid_size * ((iter + 1) % 2);

        MPI_Put(&aold[ind(1, 1)], bx, MPI_DOUBLE, north,
                ind(1, by + 1) + offset, bx, MPI_DOUBLE, win);

        MPI_Put(&aold[ind(1, by)], bx, MPI_DOUBLE, south, ind(1, 0) + offset, bx, MPI_DOUBLE, win);

        MPI_Put(&aold[ind(bx, 1)], 1, east_west_type, east,
                ind(0, 1) + offset, 1, east_west_type, win);

        MPI_Put(&aold[ind(1, 1)], 1, east_west_type, west,
                ind(bx + 1, 1) + offset, 1, east_west_type, win);

        MPI_Win_complete(win);
        MPI_Win_wait(win);      /* MEM_MODE: update to my public window becomes visible in private window */

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

    t2 = MPI_Wtime();

    MPI_Win_free(&win);
    MPI_Group_free(&neighbors_group);
    MPI_Type_free(&east_west_type);

    /* get final heat in the system */
    MPI_Allreduce(&heat, &rheat, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
        printf("[%i] last heat: %f time: %f\n", rank, rheat, t2 - t1);

    MPI_Finalize();
    return 0;
}
