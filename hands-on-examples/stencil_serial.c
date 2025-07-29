/*
 * 2D stencil code
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Input parameters */
int n;
int niters;

/* row-major order */
#define ind(i,j) ((j)*(bx+2)+(i))

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    /* get input parameters from command line options */
    if (argc != 3) {
        printf("usage: %s <n> <niters>\n", argv[0]);
        return 0;
    }

    n = atoi(argv[1]);          /* n x n grid */
    niters = atoi(argv[2]);     /* number of iterations */

    /* domain size */
    int bx, by;
    bx = n;
    by = n;

    /* energy to be injected per iteration per source */
    int energy = 1.0;

    /* initialize three heat sources */
#define NSOURCES 3
    int sources[NSOURCES][2];
    sources[0][0] = n / 2;
    sources[0][1] = n / 2;
    sources[1][0] = n / 3;
    sources[1][1] = n / 3;
    sources[2][0] = n * 4 / 5;
    sources[2][1] = n * 8 / 9;

    /* allocate working arrays & communication buffers.
     * NOTE: Include 1-wide halo zones on each side. */
    double *aold, *anew;
    anew = malloc((bx + 2) * (by + 2) * sizeof(double));
    aold = malloc((bx + 2) * (by + 2) * sizeof(double));
    /* initialize */
    memset(aold, 0, (bx + 2) * (by + 2) * sizeof(double));
    memset(anew, 0, (bx + 2) * (by + 2) * sizeof(double));

    double t_begin = MPI_Wtime();
    double last_heat;
    for (int iter = 0; iter < niters; ++iter) {
        /* refresh heat sources */
        for (int i = 0; i < NSOURCES; ++i) {
            aold[ind(sources[i][0], sources[i][1])] += energy;
        }

        /* update grid points */
        double heat = 0.0;
        for (int i = 1; i < bx + 1; ++i) {
            for (int j = 1; j < by + 1; ++j) {
                anew[ind(i, j)] = aold[ind(i, j)] / 2.0 +
                                  (aold[ind(i - 1, j)] + aold[ind(i + 1, j)] +
                                   aold[ind(i, j - 1)] + aold[ind(i, j + 1)]) / 4.0 / 2.0;
                heat += anew[ind(i, j)];
            }
        }

        last_heat = heat;

        /* swap working arrays */
        double *tmp = anew;
        anew = aold;
        aold = tmp;
    }

    double t_end = MPI_Wtime();

    /* free working arrays and communication buffers */
    free(aold);
    free(anew);

    /* get final heat in the system */
    printf("last heat: %f\n", last_heat);
    printf("    Total computation time: %.6f sec.\n", t_end - t_begin);

    MPI_Finalize();
    return 0;
}
