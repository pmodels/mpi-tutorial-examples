/* SLIDE: MPI-IO Stencil Checkpoint Code Walkthrough */
/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *
 *  (C) 2004 by University of Chicago.
 *      See CObyRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "checkpoint.h"

/* MPI-IO implementation of checkpoint and restart for MPI Stencil
 *
 * Data stored in matrix order, with a header consisting of two
 * integers: matrix size in rows (=columns), and iteration no.
 *
 * Each checkpoint is stored in its own file.
 */

static int STENCILIO_Type_create_blk(double *matrix, int bx, int by,
                                     int ghost, MPI_Datatype * newtype);

static int STENCILIO_Type_create_corner(double *matrix, int bx, int by,
                                        int ghost, MPI_Datatype * newtype);

static int STENCILIO_Type_create_filetype(int bx, int by, int cols, MPI_Datatype * newtype);

static void STENCILIO_Print_matrix(double *matrix, int bx, int by,
                                   int iter, int *coords, char *text, int rank);

/* SLIDE: MPI-IO Stencil Checkpoint Code Walkthrough */
static MPI_Comm stencilio_comm = MPI_COMM_NULL;

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
    return 1;
}

/* SLIDE: Stencil MPI-IO Checkpoint/Restart */
int STENCILIO_Checkpoint(char *prefix, double *matrix, int n,
                         int *coords, int bx, int by, int iter, MPI_Info info)
{
    int err;
    int amode = MPI_MODE_WRONLY | MPI_MODE_CREATE | MPI_MODE_UNIQUE_OPEN;
    int rank, nprocs;
    int myrows, myoffset;
    int header[2];              /* rows/cols, iteration */

    MPI_File fh;

    MPI_Datatype type;
    MPI_Datatype filetype;
    MPI_Offset myfileoffset;

    char filename[64];

    MPI_Comm_size(stencilio_comm, &nprocs);
    MPI_Comm_rank(stencilio_comm, &rank);

    /* export STENCIL_DBG_RANK and STENCIL_DBG_ITER to print rank's matrix */
    STENCILIO_Print_matrix(matrix, bx, by, iter, coords, prefix, rank);

    snprintf(filename, 63, "%s-%d.chkpt", prefix, iter);

    err = MPI_File_open(stencilio_comm, filename, amode, info, &fh);
    if (err != MPI_SUCCESS) {
        fprintf(stderr, "Error opening %s.\n", filename);
        return err;
    }

    if (rank == 0) {
        /* SLIDE: Stencil MPI-IO Checkpoint/Restart */
        header[0] = n;
        header[1] = iter;
        err = MPI_File_write_at(fh, 0, header, 2, MPI_INT, MPI_STATUS_IGNORE);
    }

    STENCILIO_Type_create_blk(matrix, bx, by, 2, &type);
    MPI_Type_commit(&type);

    myfileoffset = ((coords[1] * by) * n + coords[0] * bx) * sizeof(double) + 2 * sizeof(int);  /* include header */

    /* set file filetype */
    STENCILIO_Type_create_filetype(bx, by, n, &filetype);
    MPI_Type_commit(&filetype);
    MPI_File_set_view(fh, myfileoffset, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

    err = MPI_File_write_all(fh, MPI_BOTTOM, 1, type, MPI_STATUS_IGNORE);
    MPI_Type_free(&type);
    MPI_Type_free(&filetype);

    err = MPI_File_close(&fh);

    if (getenv("STENCIL_DBG_CORNER")) { /* export to enable checkpoint of corner data */
        snprintf(filename, 63, "%s-corner-%d.chkpt", prefix, iter);

        err = MPI_File_open(stencilio_comm, filename, amode, info, &fh);
        if (err != MPI_SUCCESS) {
            fprintf(stderr, "Error opening %s.\n", filename);
            return err;
        }

        STENCILIO_Type_create_corner(matrix, bx, by, 2, &type);
        MPI_Type_commit(&type);

        myfileoffset = rank * ((bx + 1) * 2 + (by + 1) * 2) * sizeof(double);

        /* write contiguous instead of interleaved */
        err = MPI_File_write_at_all(fh, myfileoffset, MPI_BOTTOM, 1, type, MPI_STATUS_IGNORE);
        MPI_Type_free(&type);

        err = MPI_File_close(&fh);
    }

    return err;
}

/* SLIDE: Stencil MPI-IO Checkpoint/Restart */
int STENCILIO_Restart(char *prefix, double *matrix, int n,
                      int *coords, int bx, int by, int iter, MPI_Info info)
{
    int err, gErr;
    int amode = MPI_MODE_RDONLY | MPI_MODE_UNIQUE_OPEN;
    int rank, nprocs;
    int myrows, myoffset;
    int header[2];              /* rows/cols, iteration */

    MPI_File fh;

    MPI_Datatype type;
    MPI_Datatype filetype;
    MPI_Offset myfileoffset;

    char filename[64];

    MPI_Comm_size(stencilio_comm, &nprocs);
    MPI_Comm_rank(stencilio_comm, &rank);

    snprintf(filename, 63, "%s-%d.chkpt", prefix, iter);

    err = MPI_File_open(stencilio_comm, filename, amode, info, &fh);
    if (err != MPI_SUCCESS)
        return err;

    /* check that rows and cols match */
    err = MPI_File_read_at_all(fh, 0, header, 2, MPI_INT, MPI_STATUS_IGNORE);

    /* SLIDE: Stencil MPI-IO Checkpoint/Restart */
    /* Have all process check that nothing went wrong */
    MPI_Allreduce(&err, &gErr, 1, MPI_INT, MPI_MAX, stencilio_comm);
    if (gErr || header[0] != n) {
        if (rank == 0)
            fprintf(stderr, "restart failed.\n");
        return MPI_ERR_OTHER;
    }

    STENCILIO_Type_create_blk(matrix, bx, by, 2, &type);
    MPI_Type_commit(&type);

    myfileoffset = ((coords[1] * by) * n + coords[0] * bx) * sizeof(double) + 2 * sizeof(int);  /* include header */

    /* set file filetype */
    STENCILIO_Type_create_filetype(bx, by, n, &filetype);
    MPI_Type_commit(&filetype);
    MPI_File_set_view(fh, myfileoffset, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

    err = MPI_File_read_all(fh, MPI_BOTTOM, 1, type, MPI_STATUS_IGNORE);
    MPI_Type_free(&type);
    MPI_Type_free(&filetype);

    err = MPI_File_close(&fh);

    if (getenv("STENCIL_DBG_CORNER")) { /* enable checkpoint of corner data */
        snprintf(filename, 63, "%s-corner-%d.chkpt", prefix, iter);

        err = MPI_File_open(stencilio_comm, filename, amode, info, &fh);
        if (err != MPI_SUCCESS) {
            fprintf(stderr, "Error opening %s.\n", filename);
            return err;
        }

        STENCILIO_Type_create_corner(matrix, bx, by, 2, &type);
        MPI_Type_commit(&type);

        myfileoffset = rank * ((bx + 1) * 2 + (by + 1) * 2) * sizeof(double);

        /* read contiguous instead of interleaved */
        err = MPI_File_read_at_all(fh, myfileoffset, MPI_BOTTOM, 1, type, MPI_STATUS_IGNORE);
        MPI_Type_free(&type);

        err = MPI_File_close(&fh);
    }

    /* export STENCIL_DBG_RANK and STENCIL_DBG_ITER to print rank's matrix */
    STENCILIO_Print_matrix(matrix, bx, by, iter, coords, prefix, rank);

    return err;
}

/* SLIDE: Placing Data in Checkpoint */
/* STENCILIO_Type_create_blk
 *
 * See stdio version for details (this is a copy).
 */
static int STENCILIO_Type_create_blk(double *matrix, int bx, int by,
                                     int ghost, MPI_Datatype * newtype)
{
    int err, len;
    MPI_Datatype vectype;

    MPI_Aint disp;

    /* since our data is in one block, access is very regular */
    err = MPI_Type_vector(by, bx, bx + ghost, MPI_DOUBLE, &vectype);
    if (err != MPI_SUCCESS)
        return err;

    /* wrap the vector in a type starting at the right offset */
    len = 1;
    MPI_Get_address(&matrix[(bx + 2) + 1], &disp);
    err = MPI_Type_create_hindexed(1, &len, &disp, vectype, newtype);

    MPI_Type_free(&vectype);    /* decrement reference count */

    return err;
}

static int STENCILIO_Type_create_corner(double *matrix, int bx, int by,
                                        int ghost, MPI_Datatype * newtype)
{
    int err;
    MPI_Datatype top_bottom, left_right;
    MPI_Datatype types[3];
    int lens[3] = { 1, 1, 1 };

    MPI_Aint displs[3];

    MPI_Type_contiguous(bx + 1, MPI_DOUBLE, &top_bottom);
    MPI_Get_address(&matrix[0], &displs[0]);
    MPI_Get_address(&matrix[(by + 1) * (bx + 2) + 1], &displs[1]);

    MPI_Type_vector(by + 1, 2, bx + 2, MPI_DOUBLE, &left_right);
    MPI_Get_address(&matrix[bx + 1], &displs[2]);

    /* create struct type */
    types[0] = top_bottom;
    types[1] = top_bottom;
    types[2] = left_right;

    err = MPI_Type_create_struct(3, lens, displs, types, newtype);

    return err;
}

static int STENCILIO_Type_create_filetype(int bx, int by, int cols, MPI_Datatype * newtype)
{
    return MPI_Type_vector(by, bx, cols, MPI_DOUBLE, newtype);
}

static void STENCILIO_Print_matrix(double *matrix, int bx, int by,
                                   int iter, int *coords, char *text, int rank)
{
    int i, j;
    int env_iter = -1;
    int env_proc = -1;
    char *env_iter_str;
    char *env_proc_str;

    env_proc_str = getenv("STENCIL_DBG_RANK");
    env_iter_str = getenv("STENCIL_DBG_ITER");

    if (env_proc_str) {
        env_proc = atoi(env_proc_str);
    }
    if (env_iter_str) {
        env_iter = atoi(env_iter_str);
    }

    if (rank == env_proc && (iter == env_iter || env_iter == -1)) {
        printf("%s proc %d coords=[%d,%d] iter %d:\n", text, rank, coords[0], coords[1], iter);
        for (i = 0; i < by + 2; i++) {
            for (j = 0; j < bx + 2; j++) {
                printf("%f ", matrix[(i * (bx + 2)) + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
