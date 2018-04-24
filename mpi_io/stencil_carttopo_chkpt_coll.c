/*
 * Copyright (c) 2012 Torsten Hoefler. All rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@illinois.edu>
 *
 */

/*
 * 2D stencil code with a checkpoint.
 *
 * 2D regular grid is divided into px * py blocks of grid points (px * py = # of processes.)
 * After updating grid points, each process saves the whole grid data using MPI IO operations.
 */

#include "stencil_par.h"

#define MAX_FILENAME_LENGTH (128)

void setup(int rank, int proc, int argc, char **argv,
           int *n_ptr, int *energy_ptr, int *niters_ptr, int *px_ptr, int *py_ptr,
           char **opt_prefix, int *opt_restart_iter, int *final_flag);

void init_sources(int bx, int by, int offx, int offy, int n,
                  const int nsources, int sources[][2], int *locnsources_ptr, int locsources[][2]);

void update_grid(int bx, int by, double *aold, double *anew, double *heat_ptr);

int main(int argc, char **argv)
{
    int rank, size;
    int n, energy, niters, px, py;

    int north, south, west, east;
    int bx, by, offx, offy;

    /* three heat sources */
    const int nsources = 3;
    int sources[nsources][2];
    int locnsources;            /* number of sources in my area */
    int locsources[nsources][2];        /* sources local to my rank */

    double t1, t2;

    int iter, i;

    double *aold, *anew, *tmp;

    double heat, rheat;

    int final_flag;

    /* Checkpoint/Restart support variables */
    int err;
    int amode;                  /* file access mode */
    int opt_restart_iter;       /* restart iteration */
    int header[2];              /* file header metadata */
    char *opt_prefix = NULL;
    char *old_name, *new_name;  /*aold and anew name prefix */
    char filename[MAX_FILENAME_LENGTH] = { 0 }; /* checkpoint file name */
    MPI_File fh;                /* MPI file handle */
    MPI_Datatype memtype;       /* memory layout */
    MPI_Datatype filetype;      /* file layout */
    MPI_Offset myfileoffset;

    /* initialize MPI envrionment */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* argument checking and setting */
    setup(rank, size, argc, argv, &n, &energy, &niters,
          &px, &py, &opt_prefix, &opt_restart_iter, &final_flag);

    if (final_flag == 1) {
        MPI_Finalize();
        exit(0);
    }

    /* initialize checkpoint prefix names */
    if (opt_prefix) {
        old_name = (char *) malloc(strlen(opt_prefix) + strlen("_old"));
        new_name = (char *) malloc(strlen(opt_prefix) + strlen("_new"));
        sprintf(old_name, "%s_old", opt_prefix);
        sprintf(new_name, "%s_new", opt_prefix);
    }

    /* Create a communicator with a topology */
    MPI_Comm cart_comm;
    int dims[2] = { 0, 0 };
    int periods[2] = { 0, 0 };
    int coords[2];

    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    /* determine my four neighbors */
    MPI_Cart_shift(cart_comm, 0, 1, &west, &east);
    MPI_Cart_shift(cart_comm, 1, 1, &north, &south);

    /* decompose the domain */
    bx = n / px;        /* block size in x */
    by = n / py;        /* block size in y */
    offx = coords[0] * bx;      /* offset in x */
    offy = coords[1] * by;      /* offset in y */

    /* printf("%i (%i,%i) - w: %i, e: %i, n: %i, s: %i\n", rank, ry,rx,west,east,north,south); */

    /* allocate working arrays & communication buffers */
    aold = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */
    anew = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */

    memset(aold, 0, (bx + 2) * (by + 2) * sizeof(double));
    memset(anew, 0, (bx + 2) * (by + 2) * sizeof(double));

    /* initialize three heat sources */
    init_sources(bx, by, offx, offy, n, nsources, sources, &locnsources, locsources);

    /* create east-west datatype */
    MPI_Datatype east_west_type;
    MPI_Type_vector(by, 1, bx + 2, MPI_DOUBLE, &east_west_type);
    MPI_Type_commit(&east_west_type);

    t1 = MPI_Wtime();   /* take time */

    iter = 0;

    /* create memory data layout for later I/O */
    err = MPI_Type_vector(by, bx, bx + 2, MPI_DOUBLE, &memtype);
    MPI_Type_commit(&memtype);

    /* create file data layout for later I/O */
    err = MPI_Type_vector(by, bx, n, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);

    /* compute my offset inside file, keeping into account the header (+ 2 * sizeof(int)) */
    myfileoffset = ((coords[1] * by) * n + coords[0] * bx) * sizeof(double) + 2 * sizeof(int);

    /* Check if restart is needed */
    if (opt_restart_iter > 0) {
        snprintf(filename, MAX_FILENAME_LENGTH, "%s-%d.chkpt", old_name, opt_restart_iter);

        /* open aold checkpoint file */
        amode = MPI_MODE_RDONLY | MPI_MODE_UNIQUE_OPEN;
        err = MPI_File_open(MPI_COMM_WORLD, filename, amode, MPI_INFO_NULL, &fh);
        if (err != MPI_SUCCESS) {
            fprintf(stderr, "Error opening checkpoint file %s.\n", filename);
            MPI_Abort(MPI_COMM_WORLD, err);
        }

        /* read header */
        err = MPI_File_read_at_all(fh, 0, header, 2, MPI_INT, MPI_STATUS_IGNORE);

        /* have all processes check that nothing went wrong */
        int gErr;
        MPI_Allreduce(&err, &gErr, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (gErr || header[0] != n) {
            if (rank == 0) {
                fprintf(stderr, "Restart failed.\n");
            }
            MPI_Abort(MPI_COMM_WORLD, err);
        }

        /* set file view */
        MPI_File_set_view(fh, myfileoffset, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

        /* read data */
        err = MPI_File_read_all(fh, aold + (bx + 2 + 1), 1, memtype, MPI_STATUS_IGNORE);

        /* close file */
        err = MPI_File_close(&fh);

        snprintf(filename, MAX_FILENAME_LENGTH, "%s-%d.chkpt", new_name, opt_restart_iter);

        /* open anew checkpoint file */
        err = MPI_File_open(MPI_COMM_WORLD, filename, amode, MPI_INFO_NULL, &fh);
        if (err != MPI_SUCCESS) {
            fprintf(stderr, "Error opening checkpoint file %s.\n", filename);
            MPI_Abort(MPI_COMM_WORLD, err);
        }

        /* read header */
        err = MPI_File_read_at_all(fh, 0, header, 2, MPI_INT, MPI_STATUS_IGNORE);

        /* have all processes check that nothing went wrong */
        MPI_Allreduce(&err, &gErr, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (gErr || header[0] != n) {
            if (rank == 0) {
                fprintf(stderr, "Restart failed.\n");
            }
            MPI_Abort(MPI_COMM_WORLD, err);
        }

        /* set file view */
        MPI_File_set_view(fh, myfileoffset, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

        /* read data */
        err = MPI_File_read_all(fh, anew + (bx + 2 + 1), 1, memtype, MPI_STATUS_IGNORE);

        /* close file */
        err = MPI_File_close(&fh);

        iter = opt_restart_iter + 1;
    }

    for (; iter < niters; ++iter) {

        /* refresh heat sources */
        for (i = 0; i < locnsources; ++i) {
            aold[ind(locsources[i][0], locsources[i][1])] += energy;    /* heat source */
        }

        /* exchange data with neighbors */
        MPI_Request reqs[8];
        MPI_Isend(&aold[ind(1, 1)] /* north */ , bx, MPI_DOUBLE, north, 9, cart_comm, &reqs[0]);
        MPI_Isend(&aold[ind(1, by)] /* south */ , bx, MPI_DOUBLE, south, 9, cart_comm, &reqs[1]);
        MPI_Isend(&aold[ind(bx, 1)] /* east */ , 1, east_west_type, east, 9, cart_comm, &reqs[2]);
        MPI_Isend(&aold[ind(1, 1)] /* west */ , 1, east_west_type, west, 9, cart_comm, &reqs[3]);
        MPI_Irecv(&aold[ind(1, 0)] /* north */ , bx, MPI_DOUBLE, north, 9, cart_comm, &reqs[4]);
        MPI_Irecv(&aold[ind(1, by + 1)] /* south */ , bx, MPI_DOUBLE, south, 9, cart_comm,
                  &reqs[5]);
        MPI_Irecv(&aold[ind(bx + 1, 1)] /* east */ , 1, east_west_type, east, 9, cart_comm,
                  &reqs[6]);
        MPI_Irecv(&aold[ind(0, 1)] /* west */ , 1, east_west_type, west, 9, cart_comm, &reqs[7]);
        MPI_Waitall(8, reqs, MPI_STATUS_IGNORE);

        /* update grid points */
        update_grid(bx, by, aold, anew, &heat);

        /* swap working arrays */
        tmp = anew;
        anew = aold;
        aold = tmp;

/* ==== checkpoint both aold and anew ==== */
        snprintf(filename, MAX_FILENAME_LENGTH, "%s-%d.chkpt", old_name, iter);

        /* set access mode for checkpoint file to write only */
        amode = MPI_MODE_WRONLY | MPI_MODE_CREATE | MPI_MODE_UNIQUE_OPEN;

        /* open aold checkpoint file */
        err = MPI_File_open(MPI_COMM_WORLD, filename, amode, MPI_INFO_NULL, &fh);

        /* rank 0 writes the header metadata (n, iter) */
        if (rank == 0) {
            header[0] = n;
            header[1] = iter;
            err = MPI_File_write_at(fh, 0, header, 2, MPI_INT, MPI_STATUS_IGNORE);
        }

        /* finally, set file view for checkpoint file
         * think of this as the layout of the receive process in a send/recv op */
        MPI_File_set_view(fh, myfileoffset, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

        /* write aold buffer to file */
        err = MPI_File_write_all(fh, aold + (bx + 2 + 1), 1, memtype, MPI_STATUS_IGNORE);

        /* close the file and force data to disk */
        err = MPI_File_close(&fh);

        /* update file name */
        snprintf(filename, MAX_FILENAME_LENGTH, "%s-%d.chkpt", new_name, iter);

        /* open anew checkpoint file */
        err = MPI_File_open(MPI_COMM_WORLD, filename, amode, MPI_INFO_NULL, &fh);

        /* rank 0 writes the header metadata (n, iter) */
        if (rank == 0) {
            err = MPI_File_write_at(fh, 0, header, 2, MPI_INT, MPI_STATUS_IGNORE);
        }

        /* set file view */
        MPI_File_set_view(fh, myfileoffset, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

        /* write anew buffer to file */
        err = MPI_File_write_all(fh, anew + (bx + 2 + 1), 1, memtype, MPI_STATUS_IGNORE);

        /* close the file and force data to disk */
        err = MPI_File_close(&fh);

        /* optional - print image */
        if (iter == niters - 1)
            printarr_par(iter, anew, n, px, py, coords[0], coords[1],
                         bx, by, offx, offy, MPI_COMM_WORLD);
    }

    t2 = MPI_Wtime();

    free(old_name);
    free(new_name);

    /* free working arrays and communication buffers */
    free(aold);
    free(anew);

    MPI_Type_free(&east_west_type);
    MPI_Type_free(&memtype);
    MPI_Type_free(&filetype);

    /* get final heat in the system */
    MPI_Allreduce(&heat, &rheat, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (!rank)
        printf("[%i] last heat: %f time: %f\n", rank, rheat, t2 - t1);

    MPI_Finalize();
}

void setup(int rank, int proc, int argc, char **argv,
           int *n_ptr, int *energy_ptr, int *niters_ptr, int *px_ptr, int *py_ptr,
           char **opt_prefix, int *opt_restart_iter, int *final_flag)
{
    int n, energy, niters, px, py;

    (*final_flag) = 0;

    if (argc < 6) {
        if (!rank)
            printf
                ("usage: stencil_mpi <n> <energy> <niters> <px> <py> <ckpt_prefix> <restart_iter>\n");
        (*final_flag) = 1;
        return;
    }

    n = atoi(argv[1]);  /* nxn grid */
    energy = atoi(argv[2]);     /* energy to be injected per iteration */
    niters = atoi(argv[3]);     /* number of iterations */
    px = atoi(argv[4]); /* 1st dim processes */
    py = atoi(argv[5]); /* 2nd dim processes */

    if (argc > 6 && argc <= 7) {
        *opt_prefix = argv[6];  /* checkpoint file prefix */
    } else if (argc > 7) {
        *opt_prefix = argv[6];
        *opt_restart_iter = atoi(argv[7]);      /* restart from iteration */
    }

    if (px * py != proc)
        MPI_Abort(MPI_COMM_WORLD, 1);   /* abort if px or py are wrong */
    if (n % py != 0)
        MPI_Abort(MPI_COMM_WORLD, 2);   /* abort px needs to divide n */
    if (n % px != 0)
        MPI_Abort(MPI_COMM_WORLD, 3);   /* abort py needs to divide n */

    (*n_ptr) = n;
    (*energy_ptr) = energy;
    (*niters_ptr) = niters;
    (*px_ptr) = px;
    (*py_ptr) = py;
}

void init_sources(int bx, int by, int offx, int offy, int n,
                  const int nsources, int sources[][2], int *locnsources_ptr, int locsources[][2])
{
    int i, locnsources = 0;

    sources[0][0] = n / 2;
    sources[0][1] = n / 2;
    sources[1][0] = n / 3;
    sources[1][1] = n / 3;
    sources[2][0] = n * 4 / 5;
    sources[2][1] = n * 8 / 9;

    for (i = 0; i < nsources; ++i) {    /* determine which sources are in my patch */
        int locx = sources[i][0] - offx;
        int locy = sources[i][1] - offy;
        if (locx >= 0 && locx < bx && locy >= 0 && locy < by) {
            locsources[locnsources][0] = locx + 1;      /* offset by halo zone */
            locsources[locnsources][1] = locy + 1;      /* offset by halo zone */
            locnsources++;
        }
    }

    (*locnsources_ptr) = locnsources;
}


void update_grid(int bx, int by, double *aold, double *anew, double *heat_ptr)
{
    int i, j;
    double heat = 0.0;

    for (i = 1; i < bx + 1; ++i) {
        for (j = 1; j < by + 1; ++j) {
            anew[ind(i, j)] =
                anew[ind(i, j)] / 2.0 + (aold[ind(i - 1, j)] + aold[ind(i + 1, j)] +
                                         aold[ind(i, j - 1)] + aold[ind(i, j + 1)]) / 4.0 / 2.0;
            heat += anew[ind(i, j)];
        }
    }

    (*heat_ptr) = heat;
}
