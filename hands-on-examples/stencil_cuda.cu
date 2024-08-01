/*
 * 2D stencil code
 */

#include <cuda_runtime.h>
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

__global__
void init_grid(double *anew, double *aold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    anew[i] = 0.0;
    aold[i] = 0.0;
}

__global__
void update_source(double *aold, int bx, int by, int i, int j, double energy)
{
    aold[ind(i,j)] += energy;
}

__global__
void update_grid(double *anew, double *aold, int bx, int by)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= bx && j <= by) {
        anew[ind(i, j)] = aold[ind(i, j)] / 2.0 +
                          (aold[ind(i - 1, j)] + aold[ind(i + 1, j)] +
                           aold[ind(i, j - 1)] + aold[ind(i, j + 1)]) / 4.0 / 2.0;
    }
}

__global__
void gather_heat(double *aold, int bx, int by, double *heat)
{
    *heat = 0.0;
    for (int i = 1; i < bx + 1; i++)
        for (int j = 1; j < by + 1; j++)
            *heat += aold[ind(i, j)];
}

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

    /* NOTE: use 8x8 block size. */
    assert(bx % 8 == 0);
    assert(by % 8 == 0);

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
     * NOTE: Include 1-wide hallo zones on each side. */
    double *aold, *anew;
    cudaMalloc(&anew, (bx + 2) * (by + 2) * sizeof(double));
    cudaMalloc(&aold, (bx + 2) * (by + 2) * sizeof(double));
    /* initialize */
    init_grid<<<by + 2, bx + 2>>>(anew, aold);

    /* prepare kernel launching dimesnions */
    dim3 block_dim = dim3(8, 8);
    dim3 grid_dim = dim3(bx/8, by/8);
    if (bx % 8) grid_dim.x++;
    if (by % 8) grid_dim.y++;

    double t_begin = MPI_Wtime();
    for (int iter = 0; iter < niters; ++iter) {
        /* refresh heat sources */
        for (int i = 0; i < NSOURCES; ++i) {
            update_source<<<1, 1>>>(aold, bx, by, sources[i][0], sources[i][1], energy);
        }

        /* update grid points */
        update_grid<<<grid_dim, block_dim>>>(anew, aold, bx, by);

        /* swap working arrays */
        double *tmp = anew;
        anew = aold;
        aold = tmp;
    }

    double last_heat;
    double *heat;
    cudaMalloc(&heat, sizeof(double));
    gather_heat<<<1, 1>>>(aold, bx, by, heat);
    cudaMemcpy(&last_heat, heat, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(heat);

    double t_end = MPI_Wtime();

    /* free working arrays and communication buffers */
    cudaFree(aold);
    cudaFree(anew);

    /* get final heat in the system */
    printf("last heat: %f\n", last_heat);
    printf("    Total computation time: %.6f sec.\n", t_end - t_begin);

    MPI_Finalize();
    return 0;
}
