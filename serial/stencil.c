/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * 2D stencil code
 */

/* row-major order */
#define ind(i,j) ((j)*(bx+2)+(i))
int ind_f(int i, int j, int bx)
{
    return ind(i, j);
}

/* utility functions */
void printarr(int iter, double *array, int size, int bx, int by, int (*ind) (int, int, int));
void setup(int argc, char **argv, int *n_ptr, int *energy_ptr, int *niters_ptr, int *final_flag);
void init_sources(int bx, int by, int n, const int nsources, int sources[][2]);
void alloc_bufs(int bx, int by, double **aold_ptr, double **anew_ptr);
void update_grid(int bx, int by, double *aold, double *anew, double *heat_ptr);
void free_bufs(double *aold, double *anew);

int main(int argc, char **argv)
{
    int n, energy, niters;
    int bx, by;
    /* three heat sources */
    const int nsources = 3;
    int sources[nsources][2];
    int iter, i;
    double *aold, *anew, *tmp;
    double heat;
    int final_flag;

    /* argument checking and setting */
    setup(argc, argv, &n, &energy, &niters, &final_flag);
    bx = n;
    by = n;

    if (final_flag == 1) {
        exit(0);
    }

    /* initialize three heat sources */
    init_sources(bx, by, n, nsources, sources);

    /* allocate working arrays & communication buffers */
    alloc_bufs(bx, by, &aold, &anew);

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    for (iter = 0; iter < niters; ++iter) {
        /* refresh heat sources */
        for (i = 0; i < nsources; ++i) {
            aold[ind(sources[i][0], sources[i][1])] += energy;  /* heat source */
        }

        /* update grid points */
        update_grid(bx, by, aold, anew, &heat);

        /* swap working arrays */
        tmp = anew;
        anew = aold;
        aold = tmp;

        /* optional - print image */
        if (iter == niters - 1)
            printarr(iter, anew, n, bx, by, ind_f);
    }

    clock_gettime(CLOCK_REALTIME, &end);
    float elapsed =
        ((float) (end.tv_sec - start.tv_sec) + 1.0e-9 * (double) (end.tv_nsec - start.tv_nsec));

    /* free working arrays and communication buffers */
    free_bufs(aold, anew);

    /* get final heat in the system */
    printf("last heat: %f time: %f\n", heat, elapsed);

    return 0;
}

void setup(int argc, char **argv, int *n_ptr, int *energy_ptr, int *niters_ptr, int *final_flag)
{
    int n, energy, niters;

    (*final_flag) = 0;

    if (argc < 4) {
        printf("usage: stencil <n> <energy> <niters>\n");
        (*final_flag) = 1;
        return;
    }

    n = atoi(argv[1]);  /* nxn grid */
    energy = atoi(argv[2]);     /* energy to be injected per iteration */
    niters = atoi(argv[3]);     /* number of iterations */

    (*n_ptr) = n;
    (*energy_ptr) = energy;
    (*niters_ptr) = niters;
}

void init_sources(int bx, int by, int n, const int nsources, int sources[][2])
{
    sources[0][0] = n / 2;
    sources[0][1] = n / 2;
    sources[1][0] = n / 3;
    sources[1][1] = n / 3;
    sources[2][0] = n * 4 / 5;
    sources[2][1] = n * 8 / 9;
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
