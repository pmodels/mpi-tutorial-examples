/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * 2D stencil code using a nonblocking send/receive with manual packing/unpacking.
 *
 * 2D regular grid is divided into px * py blocks of grid points (px * py = # of processes.)
 * In every iteration, each process calls nonblocking operations to exchange a halo with
 * neighbors. Grid points in a halo are packed and unpacked before and after communications.
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
                     double **sbufnorth_ptr, double **sbufsouth_ptr,
                     double **sbufeast_ptr, double **sbufwest_ptr,
                     double **rbufnorth_ptr, double **rbufsouth_ptr,
                     double **rbufeast_ptr, double **rbufwest_ptr);

void pack_data(int bx, int by, double *aold,
               double *sbufnorth, double *sbufsouth, double *sbfueast, double *sbufwest);

void unpack_data(int bx, int by, double *aold,
                 double *rbufnorth, double *rbufsouth, double *rbufeast, double *rbufwest);

void free_comm_bufs(double *sbufnorth, double *sbufsouth, double *sbufeast, double *sbufwest,
                    double *rbufnorth, double *rbufsouth, double *rbufeast, double *rbufwest);

int main(int argc, char **argv)
{
    int rank, size;
    int n, energy, niters, px, py;

    int rx, ry;
    int north, south, west, east;
    int bx, by, offx, offy;

    /* three heat sources */
    const int nsources = NSOURCES;
    int sources[NSOURCES][2];
    int locnsources;            /* number of sources in my area */
    int locsources[NSOURCES][2];        /* sources local to my rank */

    int iter;

    double *sbufnorth, *sbufsouth, *sbufeast, *sbufwest;
    double *rbufnorth, *rbufsouth, *rbufeast, *rbufwest;
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
    init_sources(bx, by, offx, offy, n, nsources, sources, &locnsources, locsources);

    /* allocate working arrays & communication buffers */
    alloc_bufs(bx, by, &aold, &anew);
    alloc_comm_bufs(bx, by, &sbufnorth, &sbufsouth, &sbufeast, &sbufwest,
                    &rbufnorth, &rbufsouth, &rbufeast, &rbufwest);

    PERF_TIMER_BEGIN(TIMER_EXEC);

    for (iter = 0; iter < niters; ++iter) {

        MPI_Request reqs[8];

        /* refresh heat sources */
        refresh_heat_source(bx, nsources, sources, energy, aold);

        PERF_TIMER_BEGIN(TIMER_COMM);
        /* pack data */
        pack_data(bx, by, aold, sbufnorth, sbufsouth, sbufeast, sbufwest);

        /* exchange data with neighbors */
        MPI_Isend(sbufnorth, bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(sbufsouth, bx, MPI_DOUBLE, south, 9, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(sbufeast, by, MPI_DOUBLE, east, 9, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(sbufwest, by, MPI_DOUBLE, west, 9, MPI_COMM_WORLD, &reqs[3]);

        MPI_Irecv(rbufnorth, bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD, &reqs[4]);
        MPI_Irecv(rbufsouth, bx, MPI_DOUBLE, south, 9, MPI_COMM_WORLD, &reqs[5]);
        MPI_Irecv(rbufeast, by, MPI_DOUBLE, east, 9, MPI_COMM_WORLD, &reqs[6]);
        MPI_Irecv(rbufwest, by, MPI_DOUBLE, west, 9, MPI_COMM_WORLD, &reqs[7]);

        MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);

        /* unpack data */
        unpack_data(bx, by, aold, rbufnorth, rbufsouth, rbufeast, rbufwest);
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
    free_comm_bufs(sbufnorth, sbufsouth, sbufeast, sbufwest,
                   rbufnorth, rbufsouth, rbufeast, rbufwest);
    free_bufs(aold, anew);

    /* get final heat in the system */
    MPI_Allreduce(&heat, &rheat, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (!rank) {
        printf("[%i] last heat: %f\n", rank, rheat);
        PERF_PRINT();
    }

    MPI_Finalize();
    return 0;
}

void alloc_comm_bufs(int bx, int by, double **sbufnorth_ptr, double **sbufsouth_ptr,
                     double **sbufeast_ptr, double **sbufwest_ptr,
                     double **rbufnorth_ptr, double **rbufsouth_ptr,
                     double **rbufeast_ptr, double **rbufwest_ptr)
{
    double *sbufnorth, *sbufsouth, *sbufeast, *sbufwest;
    double *rbufnorth, *rbufsouth, *rbufeast, *rbufwest;

    /* allocate communication buffers */
    sbufnorth = (double *) malloc(bx * sizeof(double)); /* send buffers */
    sbufsouth = (double *) malloc(bx * sizeof(double));
    sbufeast = (double *) malloc(by * sizeof(double));
    sbufwest = (double *) malloc(by * sizeof(double));
    rbufnorth = (double *) malloc(bx * sizeof(double)); /* receive buffers */
    rbufsouth = (double *) malloc(bx * sizeof(double));
    rbufeast = (double *) malloc(by * sizeof(double));
    rbufwest = (double *) malloc(by * sizeof(double));

    memset(sbufnorth, 0, bx * sizeof(double));
    memset(sbufsouth, 0, bx * sizeof(double));
    memset(sbufeast, 0, by * sizeof(double));
    memset(sbufwest, 0, by * sizeof(double));
    memset(rbufnorth, 0, bx * sizeof(double));
    memset(rbufsouth, 0, bx * sizeof(double));
    memset(rbufeast, 0, by * sizeof(double));
    memset(rbufwest, 0, by * sizeof(double));

    (*sbufnorth_ptr) = sbufnorth;
    (*sbufsouth_ptr) = sbufsouth;
    (*sbufeast_ptr) = sbufeast;
    (*sbufwest_ptr) = sbufwest;
    (*rbufnorth_ptr) = rbufnorth;
    (*rbufsouth_ptr) = rbufsouth;
    (*rbufeast_ptr) = rbufeast;
    (*rbufwest_ptr) = rbufwest;
}

void free_comm_bufs(double *sbufnorth, double *sbufsouth, double *sbufeast, double *sbufwest,
                    double *rbufnorth, double *rbufsouth, double *rbufeast, double *rbufwest)
{
    free(sbufnorth);
    free(sbufsouth);
    free(sbufeast);
    free(sbufwest);
    free(rbufnorth);
    free(rbufsouth);
    free(rbufeast);
    free(rbufwest);
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

