/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "stencil_par.h"
#include "perf_stat.h"

/* row-major order */
#define ind(i,j) ((j)*(bx+2)+(i))

void setup(int rank, int proc, int argc, char **argv,
           int *n_ptr, int *energy_ptr, int *niters_ptr, int *px_ptr, int *py_ptr, int *final_flag)
{
    int n, energy, niters, px, py;

    (*final_flag) = 0;

    if (argc != 5) {
        if (rank == 0)
            printf("usage: stencil_mpi <n> <niters> <px> <py>\n");
        (*final_flag) = 1;
        return;
    }

    energy = 1;     /* energy to be injected per iteration, hardcoded to 1 */

    n = atoi(argv[1]);  /* nxn grid */
    niters = atoi(argv[2]);     /* number of iterations */
    px = atoi(argv[3]); /* 1st dim processes */
    py = atoi(argv[4]); /* 2nd dim processes */

    if (px * py != proc) {
        fprintf(stderr, "px * py must equal to the number of processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);   /* abort if px or py are wrong */
    }
    if (n % px != 0) {
        fprintf(stderr, "grid size n must be divisible by px.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);   /* abort px needs to divide n */
    }
    if (n % py != 0) {
        fprintf(stderr, "grid size n must be divisible by py.\n");
        MPI_Abort(MPI_COMM_WORLD, 3);   /* abort py needs to divide n */
    }

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

void refresh_heat_source(int bx, int nsources, int sources[][2], int energy, double *aold_ptr)
{
    PERF_TIMER_BEGIN(TIMER_COMP);
    for (int i = 0; i < nsources; ++i) {
        aold_ptr[ind(sources[i][0], sources[i][1])] += energy;  /* heat source */
    }
    PERF_TIMER_END(TIMER_COMP);
}

void alloc_bufs(int bx, int by, double **aold_ptr, double **anew_ptr)
{
    double *aold, *anew;

    /* allocate two working arrays */
    anew = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */
    aold = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */

    memset(aold, 0, (bx + 2) * (by + 2) * sizeof(double));
    memset(anew, 0, (bx + 2) * (by + 2) * sizeof(double));

    (*aold_ptr) = aold;
    (*anew_ptr) = anew;
}

void free_bufs(double *aold, double *anew)
{
    free(aold);
    free(anew);
}

void update_grid(int bx, int by, double *aold, double *anew, double *heat_ptr)
{
    int i, j;
    double heat = 0.0;

    PERF_TIMER_BEGIN(TIMER_COMP);
    for (i = 1; i < bx + 1; ++i) {
        for (j = 1; j < by + 1; ++j) {
            anew[ind(i, j)] =
                aold[ind(i, j)] / 2.0 + (aold[ind(i - 1, j)] + aold[ind(i + 1, j)] +
                                         aold[ind(i, j - 1)] + aold[ind(i, j + 1)]) / 4.0 / 2.0;
            heat += anew[ind(i, j)];
        }
    }
    PERF_TIMER_END(TIMER_COMP);

    (*heat_ptr) = heat;
}
