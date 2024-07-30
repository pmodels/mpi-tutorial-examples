/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * 2D stencil code using a blocking collective with manual packing/unpacking.
 *
 * 2D regular grid are divided into px * py blocks of grid points (px * py = # of processes.)
 * In every iteration, each process calls a blocking collective operation to exchange a halo with
 * neighbors. Grid points in a halo region are packed and unpacked before and after the collective
 * operation.
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

void alloc_comm_bufs(int bx, int by,
                     double **exchange_sbuf_ptr, double **exchange_rbuf_ptr);

void pack_data(int bx, int by, double *aold,
               double *sbufnorth, double *sbufsouth, double *sbfueast, double *sbufwest);

void unpack_data(int bx, int by, double *aold,
                 double *rbufnorth, double *rbufsouth, double *rbufeast, double *rbufwest);

void free_comm_bufs(double *exchange_sbuf, double *exchange_rbuf);

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

    int iter, i;

    double *exchange_sbuf, *exchange_rbuf;
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

    /* allocate working arrays & communication buffers */
    alloc_bufs(bx, by, &aold, &anew);
    alloc_comm_bufs(bx, by, &exchange_sbuf, &exchange_rbuf);

    /* set offset for each halo zone in the exchange buffers */
    int ebufoff_north, ebufoff_south, ebufoff_east, ebufoff_west;
    ebufoff_north = 0;
    ebufoff_south = bx;
    ebufoff_east = 2 * bx;
    ebufoff_west = 2 * bx + by;

    /* setup parameters for alltoallv */
    int *send_sizes, *send_start_indices, *recv_sizes, *recv_start_indices;

    send_sizes = (int *) malloc(size * sizeof(int));
    send_start_indices = (int *) malloc(size * sizeof(int));
    recv_sizes = (int *) malloc(size * sizeof(int));
    recv_start_indices = (int *) malloc(size * sizeof(int));

    for (i = 0; i < size; i++) {
        if (i == north) {
            send_sizes[i] = recv_sizes[i] = bx;
            send_start_indices[i] = recv_start_indices[i] = ebufoff_north;
        } else if (i == south) {
            send_sizes[i] = recv_sizes[i] = bx;
            send_start_indices[i] = recv_start_indices[i] = ebufoff_south;
        } else if (i == east) {
            send_sizes[i] = recv_sizes[i] = by;
            send_start_indices[i] = recv_start_indices[i] = ebufoff_east;
        } else if (i == west) {
            send_sizes[i] = recv_sizes[i] = by;
            send_start_indices[i] = recv_start_indices[i] = ebufoff_west;
        } else {
            send_sizes[i] = recv_sizes[i] = 0;
        }
    }

    PERF_TIMER_BEGIN(TIMER_EXEC);

    for (iter = 0; iter < niters; ++iter) {

        /* refresh heat sources */
        refresh_heat_source(bx, locnsources, locsources, energy, aold);


        /* pack data */
        PERF_TIMER_BEGIN(TIMER_COMM);
        pack_data(bx, by, aold, &exchange_sbuf[ebufoff_north], &exchange_sbuf[ebufoff_south],
                  &exchange_sbuf[ebufoff_east], &exchange_sbuf[ebufoff_west]);

        /* exchange data with neighbors */
        MPI_Alltoallv(exchange_sbuf, send_sizes, send_start_indices, MPI_DOUBLE, exchange_rbuf,
                      recv_sizes, recv_start_indices, MPI_DOUBLE, MPI_COMM_WORLD);

        /* unpack data */
        unpack_data(bx, by, aold, &exchange_rbuf[ebufoff_north], &exchange_rbuf[ebufoff_south],
                    &exchange_rbuf[ebufoff_east], &exchange_rbuf[ebufoff_west]);
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

    /* free working arrays and communication buffers */
    free_bufs(aold, anew);
    free_comm_bufs(exchange_sbuf, exchange_rbuf);

    /* get final heat in the system */
    MPI_Allreduce(&heat, &rheat, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("[%i] last heat: %f\n", rank, rheat);
        PERF_PRINT();
    }

    free(send_sizes);
    free(send_start_indices);
    free(recv_sizes);
    free(recv_start_indices);

    MPI_Finalize();
    return 0;
}

void alloc_comm_bufs(int bx, int by,
                     double **exchange_sbuf_ptr, double **exchange_rbuf_ptr)
{
    double *exchange_sbuf, *exchange_rbuf;

    /* allocate communication buffers that holds four halo zones */
    exchange_sbuf = (double *) malloc((bx + by) * 2 * sizeof(double));
    exchange_rbuf = (double *) malloc((bx + by) * 2 * sizeof(double));

    memset(exchange_sbuf, 0, (bx + by) * 2 * sizeof(double));
    memset(exchange_rbuf, 0, (bx + by) * 2 * sizeof(double));

    (*exchange_sbuf_ptr) = exchange_sbuf;
    (*exchange_rbuf_ptr) = exchange_rbuf;
}

void free_comm_bufs(double *exchange_sbuf, double *exchange_rbuf)
{
    free(exchange_sbuf);
    free(exchange_rbuf);
}

void pack_data(int bx, int by, double *aold,
               double *sbufnorth, double *sbufsouth, double *sbufeast, double *sbufwest)
{
    int i;
    for (i = 0; i < bx; ++i)
        sbufnorth[i] = aold[ind(i + 1, 1)];     /* #1 row */
    for (i = 0; i < bx; ++i)
        sbufsouth[i] = aold[ind(i + 1, by)];    /* #(by) row */
    for (i = 0; i < by; ++i)
        sbufeast[i] = aold[ind(bx, i + 1)];     /* #(bx) col */
    for (i = 0; i < by; ++i)
        sbufwest[i] = aold[ind(1, i + 1)];      /* #1 col */
}

void unpack_data(int bx, int by, double *aold,
                 double *rbufnorth, double *rbufsouth, double *rbufeast, double *rbufwest)
{
    int i;
    for (i = 0; i < bx; ++i)
        aold[ind(i + 1, 0)] = rbufnorth[i];     /* #0 row */
    for (i = 0; i < bx; ++i)
        aold[ind(i + 1, by + 1)] = rbufsouth[i];        /* #(by+1) row */
    for (i = 0; i < by; ++i)
        aold[ind(bx + 1, i + 1)] = rbufeast[i]; /* #(bx+1) col */
    for (i = 0; i < by; ++i)
        aold[ind(0, i + 1)] = rbufwest[i];      /* #0 col */
}

