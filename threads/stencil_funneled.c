/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * 2D stencil code parallelized by multiple threads with MPI_THREAD_FUNNELED.
 *
 * 2D regular grid is divided into px * py blocks of grid points (px * py = # of processes.)
 * In every iteration, each process calls nonblocking operations with derived data types to exchange
 * grid points in a halo region with neighbors. Computation is parallelized by multiple threads.
 */

#include "stencil_par.h"
#include <omp.h>
/* row-major order */
#define ind(i,j) ((j)*(bx+2)+(i))

int ind_f(int i, int j, int bx)
{
    return ind(i, j);
}

int main(int argc, char **argv)
{
    int rank, size, provided;
    int n, energy, niters, px, py;

    int rx, ry;
    int north, south, west, east;
    int bx, by, offx, offy;

    /* three heat sources */
    int sources[NSOURCES][2];
    int locnsources;            /* number of sources in my area */
    int locsources[NSOURCES][2];        /* sources local to my rank */

    double t1, t2;

    int iter, i, j;

    double *aold, *anew, *tmp;

    double heat, rheat;

    int final_flag;


    /* initialize MPI envrionment */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED)
        MPI_Abort(MPI_COMM_WORLD, 1);

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

    /* create east-west datatype */
    MPI_Datatype east_west_type;
    MPI_Type_vector(by, 1, bx + 2, MPI_DOUBLE, &east_west_type);
    MPI_Type_commit(&east_west_type);

    t1 = MPI_Wtime();   /* take time */
#pragma omp parallel private(iter,i,j)
    {
        for (iter = 0; iter < niters; ++iter) {
#pragma omp master
            {
                /* refresh heat sources */
                refresh_heat_source(bx, locnsources, locsources, energy, aold);

                /* exchange data with neighbors */
                MPI_Request reqs[8];
                MPI_Isend(&aold[ind(1, 1)] /* north */ , bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD,
                          &reqs[0]);
                MPI_Isend(&aold[ind(1, by)] /* south */ , bx, MPI_DOUBLE, south, 9, MPI_COMM_WORLD,
                          &reqs[1]);
                MPI_Isend(&aold[ind(bx, 1)] /* east */ , 1, east_west_type, east, 9, MPI_COMM_WORLD,
                          &reqs[2]);
                MPI_Isend(&aold[ind(1, 1)] /* west */ , 1, east_west_type, west, 9, MPI_COMM_WORLD,
                          &reqs[3]);
                MPI_Irecv(&aold[ind(1, 0)] /* north */ , bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD,
                          &reqs[4]);
                MPI_Irecv(&aold[ind(1, by + 1)] /* south */ , bx, MPI_DOUBLE, south, 9,
                          MPI_COMM_WORLD, &reqs[5]);
                MPI_Irecv(&aold[ind(bx + 1, 1)] /* east */ , 1, east_west_type, east, 9,
                          MPI_COMM_WORLD, &reqs[6]);
                MPI_Irecv(&aold[ind(0, 1)] /* west */ , 1, east_west_type, west, 9, MPI_COMM_WORLD,
                          &reqs[7]);
                MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);

                heat = 0.0;
            }
#pragma omp barrier

            /* update grid points */

#pragma omp for schedule(static) reduction(+:heat)
            for (i = 1; i < bx + 1; ++i) {
                for (j = 1; j < by + 1; ++j) {
                    anew[ind(i, j)] =
                        anew[ind(i, j)] / 2.0 + (aold[ind(i - 1, j)] + aold[ind(i + 1, j)] +
                                                 aold[ind(i, j - 1)] +
                                                 aold[ind(i, j + 1)]) / 4.0 / 2.0;
                    heat += anew[ind(i, j)];
                }
            }

#pragma omp master
            {
                /* swap working arrays */
                tmp = anew;
                anew = aold;
                aold = tmp;

                /* optional - print image */
                if (iter == niters - 1)
                    printarr_par(iter, anew, n, px, py, rx, ry, bx, by, offx, offy, ind_f,
                                 MPI_COMM_WORLD);
            }
        }
    }
    t2 = MPI_Wtime();

    MPI_Type_free(&east_west_type);

    /* free working arrays and communication buffers */
    free_bufs(aold, anew);

    /* get final heat in the system */
    MPI_Allreduce(&heat, &rheat, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (!rank) {
        int nthreads = omp_get_max_threads();
#if !defined(OUTPUT_TOFILE)
        printf("n,nthreads,last_heat,time,flops\n");
        printf("%d,%d,%f,%f,%f\n", n, nthreads, rheat, t2 - t1,
               ((double) n * n * 7 * niters) / (t2 - t1));
#else
        char filename[20];
        sprintf(filename, "stencil_funneled_%d_%d", px * py, n);
        FILE *out = fopen(filename, "w");
        fprintf(out, "n,nthreads,last_heat,time,flops\n");
        fprintf(out, "%d,%d,%f,%f,%f\n", n, nthreads, rheat, t2 - t1,
                ((double) n * n * 7 * niters) / (t2 - t1));
        fclose(out);
#endif
    }

    MPI_Finalize();
    return 0;
}
