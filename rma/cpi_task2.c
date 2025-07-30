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
#include <assert.h>

double f(double);

double f(double a)
{
    return sqrt(1.0 - a * a);
}

int main(int argc, char *argv[])
{
    int n = 0, myid, numprocs, i;
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

    MPI_Win w_n, w_mypi;

    if (myid == 0) {
        MPI_Win_create(&n, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &w_n);
        n = 10000;  /* default # of rectangles */
        startwtime = MPI_Wtime();
    } else {
        MPI_Win_create(NULL, 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &w_n);
    }
    MPI_Win_create(&mypi, sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &w_mypi);

    MPI_Win_fence(0, w_n);
    if (myid != 0){
        MPI_Get(&n, 1, MPI_INT, 0, 0, 1, MPI_INT, w_n);
    }
    MPI_Win_fence(0, w_n);

    h = 1.0 / (double) n;
    sum = 0.0;
    /* A slightly better approach starts from large i and works back */
    for (i = myid + 1; i <= n; i += numprocs) {
        x = h * ((double) i - 0.5);
        sum += f(x);
    }
    mypi = h * sum;

    double tmp = 0;
    pi = 0;
    MPI_Win_fence(0, w_mypi);
    if (myid != 0) {
        MPI_Accumulate(&mypi, 1, MPI_DOUBLE, 0, 0, 1, MPI_DOUBLE, MPI_SUM, w_mypi);
    }
    MPI_Win_fence(0, w_mypi);
    pi = mypi;

    pi *= 4.0;
    if (myid == 0) {
        endwtime = MPI_Wtime();
        printf("pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));
        printf("wall clock time = %f\n", endwtime - startwtime);
        fflush(stdout);
    }

    MPI_Win_free(&w_n);
    MPI_Win_free(&w_mypi);

    MPI_Finalize();
    return 0;
}
