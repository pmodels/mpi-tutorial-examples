/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * Calculation of Pi via numerical integration.
 *
 * Parallelly integrate sqrt(1 - x^2) over x from 0 to 1 by dividing x with multiple nodes.
 * The result is reduced by a blocking collective.
 */

#include "mpi.h"
#include <stdio.h>
#include <math.h>

double f(double);

double f(double a)
{
    return sqrt(1.0 - a * a);
}

int main(int argc, char *argv[])
{
    int n, myid, numprocs, i;
    double PI25DT = 3.141592653589793238462643;
    double mypi, pi, h, sum, x;
    double startwtime = 0.0, endwtime;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(processor_name, &namelen);

    fprintf(stdout, "Process %d of %d is on %s\n", myid, numprocs, processor_name);
    fflush(stdout);

    /* TODO: Expose n and mypi through windows */
    /* Related functions:
     * int MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info,
                  MPI_Comm comm, MPI_Win *win) */

    if (myid == 0) {
        n = 10000;  /* default # of rectangles */
        startwtime = MPI_Wtime();
    }

    /* TODO: Replace MPI_Bcast with RMA */
    /* Related functions:
     * int MPI_Win_fence(int assert, MPI_Win win)
     * int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype
            origin_datatype, int target_rank, MPI_Aint target_disp,
            int target_count, MPI_Datatype target_datatype, MPI_Win
            win) */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    h = 1.0 / (double) n;
    sum = 0.0;
    /* A slightly better approach starts from large i and works back */
    for (i = myid + 1; i <= n; i += numprocs) {
        x = h * ((double) i - 0.5);
        sum += f(x);
    }
    mypi = h * sum;

    /* TODO: Replace MPI_Reduce with RMA and local summing loop */
    MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    pi *= 4.0;
    if (myid == 0) {
        endwtime = MPI_Wtime();
        printf("pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));
        printf("wall clock time = %f\n", endwtime - startwtime);
        fflush(stdout);
    }

    /* TODO: Cleanup windows */
    /* Related functions:
     * int MPI_Win_free(MPI_Win *win); */

    MPI_Finalize();
    return 0;
}
