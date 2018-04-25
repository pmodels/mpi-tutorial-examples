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

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include "stencil_par.h"

#define MAX_FILENAME_LENGTH (128)

/* row-major order */
#define ind(i,j) ((j)*(bx+2)+(i))

int ind_f(int i, int j, int bx)
{
    return ind(i, j);
}

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
    int opt_restart_iter;       /* restart iteration */
    int header[2];              /* file header metadata */
    int row;
    int fd;                     /* file descriptor */
    off_t stride;               /* stride inside file */
    off_t offset;               /* offset inside file */
    double *buf;                /* rank 0 recv buffer */
    char *opt_prefix = NULL;
    char *old_name, *new_name;  /*aold and anew name prefix */
    char filename[MAX_FILENAME_LENGTH] = { 0 }; /* checkpoint file name */
    MPI_Datatype memtype;
    struct coord_array {
        int xcoord;
        int ycoord;
    } *coord_array;

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

    /* send all coords to rank0 */
    if (rank == 0) {
        coord_array = (struct coord_array *) malloc(sizeof(struct coord_array) * size);
        MPI_Gather(coords, 2, MPI_INT, coord_array, 2, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(coords, 2, MPI_INT, coord_array, 2, MPI_INT, 0, MPI_COMM_WORLD);
    }

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

    if (opt_restart_iter > 0) {
        if (rank == 0) {
            buf = (double *) malloc(sizeof(double) * bx * by);

            snprintf(filename, MAX_FILENAME_LENGTH, "%s-%d.chkpt", old_name, opt_restart_iter);

            /* open aold checkpoint file */
            fd = open(filename, O_RDONLY, S_IRWXU);
            if (fd < 0) {
                fprintf(stderr, "Error opening checkpoint file %s: %s.\n", filename,
                        strerror(errno));
                MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
            }

            /* read header metadata */
            header[0] = header[1] = 0;
            pread(fd, header, 2 * sizeof(int), 0);
            if (header[1] != opt_restart_iter) {
                fprintf(stderr, "Error restarting iter %d from %s.\n", header[1], filename);
                MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
            }

            /* initialize offset and stride */
            offset = 2 * sizeof(int);
            stride = n * sizeof(double);

            /* rank 0 reads first (no comm needed) */
            for (row = 1; row <= by; row++) {
                pread(fd, aold + (bx + 2) * row + 1, bx * sizeof(double), offset);
                offset += stride;
            }

            /* read data on behalf of other ranks and send it to them */
            for (i = 1; i < size; i++) {
                offset =
                    (coord_array[i].ycoord * by * n + coord_array[i].xcoord * bx) * sizeof(double) +
                    2 * sizeof(int);
                for (row = 0; row < by; row++) {
                    pread(fd, buf + (row * bx), bx * sizeof(double), offset);
                    offset += stride;
                }
                MPI_Send(buf, bx * by, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }

            /* close checkpoint file */
            close(fd);

            snprintf(filename, MAX_FILENAME_LENGTH, "%s-%d.chkpt", new_name, opt_restart_iter);

            /* open anew checkpoint file */
            fd = open(filename, O_RDONLY, S_IRWXU);
            if (fd < 0) {
                fprintf(stderr, "Error opening checkpoint file %s: %s.\n", filename,
                        strerror(errno));
                MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
            }

            /* read header metadata */
            header[0] = header[1] = 0;
            pread(fd, header, 2 * sizeof(int), 0);
            if (header[1] != opt_restart_iter) {
                fprintf(stderr, "Error restarting iter %d from %s.\n", header[1], filename);
                MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
            }

            /* initialize offset and stride */
            offset = 2 * sizeof(int);

            /* rank 0 reads first (no comm needed) */
            for (row = 1; row <= by; row++) {
                pread(fd, anew + (bx + 2) * row + 1, bx * sizeof(double), offset);
                offset += stride;
            }

            /* read data on behalf of other ranks and send it to them */
            for (i = 1; i < size; i++) {
                offset =
                    (coord_array[i].ycoord * by * n + coord_array[i].xcoord * bx) * sizeof(double) +
                    2 * sizeof(int);
                for (row = 0; row < by; row++) {
                    pread(fd, buf + (row * bx), bx * sizeof(double), offset);
                    offset += stride;
                }
                MPI_Send(buf, bx * by, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
            }

            /* close checkpoint file */
            close(fd);

            /* free I/O buf */
            free(buf);
        } else {
            /* create memory layout */
            MPI_Type_vector(by, bx, bx + 2, MPI_DOUBLE, &memtype);
            MPI_Type_commit(&memtype);

            /* recv aold data from rank 0 */
            MPI_Recv(aold + (bx + 2 + 1), 1, memtype, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* recv anew data from rank 0 */
            MPI_Recv(anew + (bx + 2 + 1), 1, memtype, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* free memory layout object */
            MPI_Type_free(&memtype);
        }

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

        /* receive data from each process and write it to file
         * NOTE: this is not the most efficient way of writing to a file
         *       but simplifies the code and serves as example of legacy
         *       I/O strategy using POSIX. A better implementation would
         *       write data to the file sequentially instead of strided.*/
        if (rank == 0) {
            buf = (double *) malloc(sizeof(double) * bx * by);

            snprintf(filename, MAX_FILENAME_LENGTH, "%s-%d.chkpt", old_name, iter);

            /* open aold checkpoint file */
            fd = open(filename, O_CREAT | O_WRONLY, S_IRWXU);

            /* write header metadata */
            header[0] = n;
            header[1] = iter;
            pwrite(fd, header, 2 * sizeof(int), 0);

            /* initialize offset and stride */
            offset = 2 * sizeof(int);
            stride = n * sizeof(double);

            /* rank 0 writes first (no comm needed) */
            for (row = 1; row <= by; row++) {
                pwrite(fd, aold + (bx + 2) * row + 1, bx * sizeof(double), offset);
                offset += stride;
            }

            /* write data received from other ranks */
            for (i = 1; i < size; i++) {
                MPI_Recv(buf, bx * by, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset =
                    (coord_array[i].ycoord * by * n + coord_array[i].xcoord * bx) * sizeof(double) +
                    2 * sizeof(int);
                for (row = 0; row < by; row++) {
                    pwrite(fd, buf + (row * bx), bx * sizeof(double), offset);
                    offset += stride;
                }
            }

            /* close checkpoint file */
            close(fd);

            snprintf(filename, MAX_FILENAME_LENGTH, "%s-%d.chkpt", new_name, iter);

            /* open anew checkpoint file */
            fd = open(filename, O_CREAT | O_WRONLY, S_IRWXU);

            /* write header metadata */
            pwrite(fd, header, 2 * sizeof(int), 0);

            /* initialize offset */
            offset = 2 * sizeof(int);

            /* rank 0 writes first (no comm needed) */
            for (row = 1; row <= by; row++) {
                pwrite(fd, anew + (bx + 2) * row + 1, bx * sizeof(double), offset);
                offset += stride;
            }

            /* write data received from other ranks */
            for (i = 1; i < size; i++) {
                MPI_Recv(buf, bx * by, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset =
                    (coord_array[i].ycoord * by * n + coord_array[i].xcoord * bx) * sizeof(double) +
                    2 * sizeof(int);
                for (row = 0; row < by; row++) {
                    pwrite(fd, buf + (row * bx), bx * sizeof(double), offset);
                    offset += stride;
                }
            }

            /* close checkpoint file */
            close(fd);

            /* free I/O buf */
            free(buf);
        } else {
            /* create memory layout */
            MPI_Type_vector(by, bx, bx + 2, MPI_DOUBLE, &memtype);
            MPI_Type_commit(&memtype);

            /* send aold data to rank 0 */
            MPI_Send(aold + (bx + 2 + 1), 1, memtype, 0, 0, MPI_COMM_WORLD);

            /* send anew data to rank 0 */
            MPI_Send(anew + (bx + 2 + 1), 1, memtype, 0, 1, MPI_COMM_WORLD);

            /* free memory layout object */
            MPI_Type_free(&memtype);
        }

        /* optional - print image */
        if (iter == niters - 1)
            printarr_par(iter, anew, n, px, py, coords[0], coords[1],
                         bx, by, offx, offy, ind_f, MPI_COMM_WORLD);
    }

    t2 = MPI_Wtime();

    free(old_name);
    free(new_name);

    /* free working arrays and communication buffers */
    free(aold);
    free(anew);

    if (rank == 0)
        free(coord_array);

    MPI_Type_free(&east_west_type);

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
