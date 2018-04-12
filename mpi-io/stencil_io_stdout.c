/* SLIDE: stdio Stencil Checkpoint Code Walkthrough */
/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2004 by University of Chicago.
 *      See CObyRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <mpi.h>

#include "stencil_io.h"

/* stdout implementation of checkpoint (no restart) for MPI Stencil */

static int STENCILIO_Type_create_blk(double *matrix, int bx, int by,
                                     int ghost, MPI_Datatype * newtype);

static void STENCILIO_Blk_print(const double *data, int bx, int by, int rank, int ghost);

static void STENCILIO_msleep(int msec);

static MPI_Comm stencilio_comm = MPI_COMM_NULL;

/* SLIDE: stdio Stencil Checkpoint Code Walkthrough */
int STENCILIO_Init(MPI_Comm comm)
{
    int err;

    err = MPI_Comm_dup(comm, &stencilio_comm);

    return err;
}

int STENCILIO_Finalize(void)
{
    int err;

    err = MPI_Comm_free(&stencilio_comm);

    return err;
}

int STENCILIO_Can_restart(void)
{
    return 0;
}

/* SLIDE: Stencil stdout "checkpoint" */
/* STENCILIO_Checkpoint
 *
 * Parameters:
 * prefix - prefix of file to hold checkpoint (ignored)
 * matrix - data values
 * n      - domain size in each direction
 * coords - process coordinates in the domain
 * bx     - number of points along x
 * by     - number of points along y
 * iter   - iteration number of checkpoint
 * info   - hints for I/O (ignored)
 *
 * Returns MPI_SUCCESS on success, MPI error code on error.
 */
int STENCILIO_Checkpoint(char *prefix, double *matrix, int n,
                         int *coords, int bx, int by, int iter, MPI_Info info)
{
    int err = MPI_SUCCESS, rank, nprocs;
    double *data = NULL;
    MPI_Datatype type;

    MPI_Comm_size(stencilio_comm, &nprocs);
    MPI_Comm_rank(stencilio_comm, &rank);

    /* communicate matrix */
    if (rank != 0) {
        /* send all data to rank 0 */
        STENCILIO_Type_create_blk(matrix, bx, by, 2, &type);
        MPI_Type_commit(&type);
        err = MPI_Send(MPI_BOTTOM, 1, type, 0, 1, stencilio_comm);
        MPI_Type_free(&type);
        /* SLIDE: Describing Data */
    } else {
        int i;

        printf("# Iteration %d\n", iter);

        /* print rank 0 data first */
        STENCILIO_Blk_print(&matrix[(bx + 2) + 1], bx, by, rank, 2);

        /* allocate memory to receive others' data */
        MPI_Alloc_mem(by * bx * sizeof(double), MPI_INFO_NULL, &data);
        memset(data, 0, by * bx * sizeof(double));

        /* receive and print others' data */
        for (i = 1; i < nprocs; i++) {
            err = MPI_Recv(data, by * bx, MPI_DOUBLE, i, 1, stencilio_comm, MPI_STATUS_IGNORE);

            printf("Patch ");
            STENCILIO_Blk_print(data, bx, by, i, 0);
        }

        /* free memory */
        MPI_Free_mem(data);
    }

    /* SLIDE: Describing Data */

    STENCILIO_msleep(250);      /* give time to see the results */

    return err;
}

/* SLIDE: Describing Data */
/* STENCILIO_Type_create_rowblk
 *
 * Creates a MPI_Datatype describing the block of by of data
 * for the local process, not including the surrounding boundary
 * cells.
 *
 * Note: This implementation assumes that the data for matrix is
 *       allocated as one large contiguous block!
 */
static int STENCILIO_Type_create_blk(double *matrix, int bx, int by,
                                     int ghost, MPI_Datatype * newtype)
{
    int err, len;
    MPI_Datatype vectype;
    MPI_Aint disp;

    /* since our data is in one block, access is very regular! */
    err = MPI_Type_vector(by, bx, bx + 2, MPI_DOUBLE, &vectype);
    if (err != MPI_SUCCESS)
        return err;

    /* wrap the vector in a type starting at the right offset */
    len = 1;
    MPI_Get_address(&matrix[(bx + ghost) + 1], &disp);
    err = MPI_Type_create_hindexed(1, &len, &disp, vectype, newtype);

    MPI_Type_free(&vectype);    /* decrement reference count */

    return err;
}

static void STENCILIO_Blk_print(const double *data, int bx, int by, int proc, int ghost)
{
    int i, j;

    printf("proc %d: \n", proc);

    for (i = 0; i < by; i++) {
        for (j = 0; j < bx; j++) {
            printf("%f ", data[(i * (bx + ghost)) + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int STENCILIO_Restart(char *prefix, double *matrix, int n,
                      int *coords, int bx, int by, int iter, MPI_Info info)
{
    return MPI_ERR_IO;
}

#ifdef HAVE_NANOSLEEP
#include <time.h>
static void STENCILIO_msleep(int msec)
{
    struct timespec t;

    t.tv_sec = msec / 1000;
    t.tv_nsec = 1000000 * (msec - t.tv_sec);

    nanosleep(&t, NULL);
}
#else
static void STENCILIO_msleep(int msec)
{
    if (msec < 1000) {
        sleep(1);
    } else {
        sleep(msec / 1000);
    }
}
#endif
