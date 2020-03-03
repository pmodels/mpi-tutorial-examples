/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * 2D stencil code using a nonblocking send/receive with manual packing/unpacking.
 *
 * 2D regular grid is divided into px * py blocks of grid points (px * py = # of processes.)
 * In every iteration, each process calls nonblocking operations to exchange a halo with
 * neighbors. Grid points in a halo are packed and unpacked before and after communications.
 */

#include "stencil_par.h"
#include <cuda.h>
#include <cuda_runtime.h>

/* Comment out if MPI library does not support GPUDirect */
#define HAVE_GPUDIRECT

/* CUDA allows a maximum of 1024 threads per block */
#define CUDA_BLKSIZE_X 32
#define CUDA_BLKSIZE_Y 32

#define MIN(a, b) ((a < b) ? a : b)

/* row-major order */
#define ind(i,j) ((j)*(bx+2)+(i))

void setup(int rank, int proc, int argc, char **argv, int *n_ptr, int *energy_ptr, int *niters_ptr,
           int *px_ptr, int *py_ptr, int *final_flag);

void init_sources(int bx, int by, int offx, int offy, int n, const int nsources, int sources[][2],
                  int *locnsources_ptr, int locsources[][2]);

void alloc_bufs(int bx, int by, double **aold_ptr, double **anew_ptr, double **sbufnorth_ptr,
                double **sbufsouth_ptr, double **sbufeast_ptr, double **sbufwest_ptr,
                double **rbufnorth_ptr, double **rbufsouth_ptr, double **rbufeast_ptr,
                double **rbufwest_ptr);

void alloc_dev_bufs(int bx, int by, double **aold_ptr, double **anew_ptr, double **sbufnorth_ptr,
                    double **sbufsouth_ptr, double **sbufeast_ptr, double **sbufwest_ptr,
                    double **rbufnorth_ptr, double **rbufsouth_ptr, double **rbufeast_ptr,
                    double **rbufwest_ptr);

void free_bufs(double *aold, double *anew, double *sbufnorth, double *sbufsouth, double *sbufeast,
               double *sbufwest, double *rbufnorth, double *rbufsouth, double *rbufeast,
               double *rbufwest);

void free_dev_bufs(double *aold, double *anew, double *sbufnorth, double *sbufsouth,
                   double *sbufeast, double *sbufwest, double *rbufnorth, double *rbufsouth,
                   double *rbufeast, double *rbufwest);

__global__ void update_sources(int bx, int by, double heat, int nsources, int *locsources,
                               double *aold);

__global__ void pack_data(int bx, int by, double *aold, double *sbufnorth, double *sbufsouth,
                          double *sbfueast, double *sbufwest);

__global__ void unpack_data(int bx, int by, double *aold, double *rbufnorth, double *rbufsouth,
                            double *rbufeast, double *rbufwest);

__global__ void update_grid(int bx, int by, double *aold, double *anew, double *heat_ptr);

int main(int argc, char **argv)
{
    int rank, size;
    int n, energy, niters, px, py;

    int rx, ry;
    int north, south, west, east;
    int bx, by, offx, offy;

    /* three heat sources */
    const int nsources = 3;
    int sources[nsources][2];
    int locnsources;            /* number of sources in my area */
    int locsources[nsources][2];        /* sources local to my rank */
    int *locsources_d;

    double t1, t2;

    int iter, i;

#ifndef HAVE_GPUDIRECT
    /* host buffers */
    double *sbufnorth, *sbufsouth, *sbufeast, *sbufwest;
    double *rbufnorth, *rbufsouth, *rbufeast, *rbufwest;
    double *aold, *anew;
#endif

    /* device buffers */
    double *sbufnorth_d, *sbufsouth_d, *sbufeast_d, *sbufwest_d;
    double *rbufnorth_d, *rbufsouth_d, *rbufeast_d, *rbufwest_d;
    double *aold_d, *anew_d, *tmp;

    double heat = 0.0, rheat;
    double *heat_d;

    int final_flag;

    /* initialize MPI envrionment */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* argument checking and setting */
    setup(rank, size, argc, argv, &n, &energy, &niters, &px, &py, &final_flag);

    if (final_flag == 1) {
        MPI_Finalize();
        exit(0);
    }

    /* determine my coordinates (x,y) -- rank=x*a+y in the 2d processor array */
    rx = rank % px;
    ry = rank / px;

    /* determine my four neighbors */
    north = (ry - 1) * px + rx;
    if (ry - 1 < 0)
        north = MPI_PROC_NULL;
    south = (ry + 1) * px + rx;
    if (ry + 1 >= py)
        south = MPI_PROC_NULL;
    west = ry * px + rx - 1;
    if (rx - 1 < 0)
        west = MPI_PROC_NULL;
    east = ry * px + rx + 1;
    if (rx + 1 >= px)
        east = MPI_PROC_NULL;

    /* decompose the domain */
    bx = n / px;        /* block size in x */
    by = n / py;        /* block size in y */
    offx = rx * bx;     /* offset in x */
    offy = ry * by;     /* offset in y */

    /* printf("%i (%i,%i) - w: %i, e: %i, n: %i, s: %i\n", rank, ry,rx,west,east,north,south); */

    /* initialize three heat sources */
    init_sources(bx, by, offx, offy, n, nsources, sources, &locnsources, locsources);

    /* assign processes to devices */
    int local_rank, dev_id, dev_count;
    CUdevice cuDevice;
    CUcontext cuContext;
    MPI_Comm intranode_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &intranode_comm);
    MPI_Comm_rank(intranode_comm, &local_rank);

    cudaGetDeviceCount(&dev_count);
    dev_id = local_rank % dev_count;
    cudaSetDevice(dev_id);

    cuInit(0);
    cuDeviceGet(&cuDevice, dev_id);
    cuDevicePrimaryCtxRetain(&cuContext, cuDevice);
    MPI_Comm_free(&intranode_comm);

    /* create an asynchronous cuda stream */
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

    /* allocate space for params in device memory */
    cudaMalloc(&locsources_d, sizeof(int) * nsources * 2);

    /* move parameters to device memory */
    cudaMemcpy(locsources_d, locsources, sizeof(int) * nsources * 2, cudaMemcpyHostToDevice);

#ifndef HAVE_GPUDIRECT
    /* allocate working arrays & communication buffers */
    alloc_bufs(bx, by, &aold, &anew,
               &sbufnorth, &sbufsouth, &sbufeast, &sbufwest,
               &rbufnorth, &rbufsouth, &rbufeast, &rbufwest);
#endif

    /* allocate working arrays & communication buffers for device */
    alloc_dev_bufs(bx, by, &aold_d, &anew_d,
                   &sbufnorth_d, &sbufsouth_d, &sbufeast_d, &sbufwest_d,
                   &rbufnorth_d, &rbufsouth_d, &rbufeast_d, &rbufwest_d);

    /* cuda kernels execution configuration parameters */
    int cuGrdSzX, cuGrdSzY, cuBlkSzX, cuBlkSzY;
    int cuGrdSzPackUnpack, cuBlkSzPackUnpack;

    /* workout number of threads in block for each dimension */
    cuBlkSzX = (bx > CUDA_BLKSIZE_X) ? CUDA_BLKSIZE_X : bx;
    cuBlkSzY = (by > CUDA_BLKSIZE_Y) ? CUDA_BLKSIZE_Y : by;
    dim3 cuBlkSzUpdate(cuBlkSzX, cuBlkSzY);

    /* workout the size of shared memory for heat reduction */
    size_t shmSize = (cuBlkSzX * cuBlkSzY * sizeof(double));

    /* workout number of blocks in grid for each dimension */
    cuGrdSzX = (bx + (cuBlkSzX - 1)) / cuBlkSzX;
    cuGrdSzY = (by + (cuBlkSzY - 1)) / cuBlkSzY;
    dim3 cuGrdSzUpdate(cuGrdSzX, cuGrdSzY);

    /* workout number of threads in block for each pack/unpack */
    int bmax = (bx > by) ? bx : by;
    cuBlkSzPackUnpack =
        (bmax > (CUDA_BLKSIZE_X * CUDA_BLKSIZE_Y)) ? (CUDA_BLKSIZE_X * CUDA_BLKSIZE_Y) : bmax;

    /* workout number of blocks in grid for each pack/unpack */
    cuGrdSzPackUnpack = (bmax + (cuBlkSzPackUnpack - 1)) / cuBlkSzPackUnpack;

    /* allocate heat vector for reduction: threads inside the same block update
     * heat values in shared memory and reflect those updates in global memory,
     * eventually.
     *
     * Rationale: heat values cannot be reduced directly by the same kernel as
     * thread synchronization across thread blocks is not easy task. For this
     * reason we use a temp array to store partial heat updates and run the
     * reduction in the host at the end of the program instead. */
    cudaMalloc(&heat_d, sizeof(double) * cuGrdSzX * cuGrdSzY);
    cudaMemset(heat_d, 0, sizeof(double) * cuGrdSzX * cuGrdSzY);

    t1 = MPI_Wtime();   /* take time */

    for (iter = 0; iter < niters; ++iter) {

        /* refresh heat sources */
        update_sources <<< cuGrdSzUpdate, cuBlkSzUpdate, 0, s >>> (bx, by, energy, nsources,
                                                                   locsources_d, aold_d);

        /* pack data in device */
        pack_data <<< cuGrdSzPackUnpack, cuBlkSzPackUnpack, 0, s >>> (bx, by, aold_d,
                                                                      sbufnorth_d, sbufsouth_d,
                                                                      sbufeast_d, sbufwest_d);

        /* wait for stream operations to complete */
        cudaStreamSynchronize(s);

        MPI_Request reqs[8];

#ifdef HAVE_GPUDIRECT
        /* exchange data with neighbors */
        MPI_Isend(sbufnorth_d, bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(sbufsouth_d, bx, MPI_DOUBLE, south, 9, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(sbufeast_d, by, MPI_DOUBLE, east, 9, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(sbufwest_d, by, MPI_DOUBLE, west, 9, MPI_COMM_WORLD, &reqs[3]);

        MPI_Irecv(rbufnorth_d, bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD, &reqs[4]);
        MPI_Irecv(rbufsouth_d, bx, MPI_DOUBLE, south, 9, MPI_COMM_WORLD, &reqs[5]);
        MPI_Irecv(rbufeast_d, by, MPI_DOUBLE, east, 9, MPI_COMM_WORLD, &reqs[6]);
        MPI_Irecv(rbufwest_d, by, MPI_DOUBLE, west, 9, MPI_COMM_WORLD, &reqs[7]);

        MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
#else
        /* move data to host */
        cudaMemcpy(sbufnorth, sbufnorth_d, sizeof(double) * bx, cudaMemcpyDeviceToHost);
        cudaMemcpy(sbufsouth, sbufsouth_d, sizeof(double) * bx, cudaMemcpyDeviceToHost);
        cudaMemcpy(sbufeast, sbufeast_d, sizeof(double) * bx, cudaMemcpyDeviceToHost);
        cudaMemcpy(sbufwest, sbufwest_d, sizeof(double) * bx, cudaMemcpyDeviceToHost);

        /* exchange data with neighbors */
        MPI_Isend(sbufnorth, bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(sbufsouth, bx, MPI_DOUBLE, south, 9, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(sbufeast, by, MPI_DOUBLE, east, 9, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(sbufwest, by, MPI_DOUBLE, west, 9, MPI_COMM_WORLD, &reqs[3]);

        MPI_Irecv(rbufnorth, bx, MPI_DOUBLE, north, 9, MPI_COMM_WORLD, &reqs[4]);
        MPI_Irecv(rbufsouth, bx, MPI_DOUBLE, south, 9, MPI_COMM_WORLD, &reqs[5]);
        MPI_Irecv(rbufeast, by, MPI_DOUBLE, east, 9, MPI_COMM_WORLD, &reqs[6]);
        MPI_Irecv(rbufwest, by, MPI_DOUBLE, west, 9, MPI_COMM_WORLD, &reqs[7]);

        MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);

        /* move received data to device */
        cudaMemcpy(rbufnorth_d, rbufnorth, sizeof(double) * bx, cudaMemcpyHostToDevice);
        cudaMemcpy(rbufsouth_d, rbufsouth, sizeof(double) * bx, cudaMemcpyHostToDevice);
        cudaMemcpy(rbufeast_d, rbufeast, sizeof(double) * bx, cudaMemcpyHostToDevice);
        cudaMemcpy(rbufwest_d, rbufwest, sizeof(double) * bx, cudaMemcpyHostToDevice);
#endif

        /* unpack data in device */
        unpack_data <<< cuGrdSzPackUnpack, cuBlkSzPackUnpack, 0, s >>> (bx, by,
                                                                        aold_d, rbufnorth_d,
                                                                        rbufsouth_d, rbufeast_d,
                                                                        rbufwest_d);

        /* update grid points */
        update_grid <<< cuGrdSzUpdate, cuBlkSzUpdate, shmSize, s >>> (bx, by, aold_d, anew_d,
                                                                      heat_d);

        /* swap working arrays */
        tmp = anew_d;
        anew_d = aold_d;
        aold_d = tmp;
    }

    /* wait for kernel updates to grid */
    cudaStreamSynchronize(s);

    t2 = MPI_Wtime();   /* take time */

    /* reduce partial heat updates in host */
    double *heat_h = (double *) malloc(sizeof(double) * cuGrdSzX * cuGrdSzY);
    cudaMemcpy(heat_h, heat_d, sizeof(double) * cuGrdSzX * cuGrdSzY, cudaMemcpyDeviceToHost);
    for (i = 0; i < cuGrdSzX * cuGrdSzY; i++)
        heat += heat_h[i];

    /* get final heat in the system */
    MPI_Allreduce(&heat, &rheat, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (!rank)
        printf("[%i] last heat: %f time: %f\n", rank, rheat, t2 - t1);

#ifndef HAVE_GPUDIRECT
    /* free working arrays and communication buffers */
    free_bufs(aold, anew, sbufnorth, sbufsouth, sbufeast, sbufwest,
              rbufnorth, rbufsouth, rbufeast, rbufwest);
#endif

    /* free working device arrays and communication buffers */
    free_dev_bufs(aold_d, anew_d, sbufnorth_d, sbufsouth_d, sbufeast_d, sbufwest_d,
                  rbufnorth_d, rbufsouth_d, rbufeast_d, rbufwest_d);

    /* free parameters in device memory */
    cudaFree(locsources_d);
    cudaFree(heat_d);
    free(heat_h);

    cudaStreamDestroy(s);
    cuDevicePrimaryCtxRelease(cuDevice);

    MPI_Finalize();
    return 0;
}

void setup(int rank, int proc, int argc, char **argv,
           int *n_ptr, int *energy_ptr, int *niters_ptr, int *px_ptr, int *py_ptr, int *final_flag)
{
    int n, energy, niters, px, py;

    (*final_flag) = 0;

    if (argc < 6) {
        if (!rank)
            printf("usage: stencil_mpi <n> <energy> <niters> <px> <py>\n");
        (*final_flag) = 1;
        return;
    }

    n = atoi(argv[1]);  /* nxn grid */
    energy = atoi(argv[2]);     /* energy to be injected per iteration */
    niters = atoi(argv[3]);     /* number of iterations */
    px = atoi(argv[4]); /* 1st dim processes */
    py = atoi(argv[5]); /* 2nd dim processes */

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

void alloc_bufs(int bx, int by, double **aold_ptr, double **anew_ptr,
                double **sbufnorth_ptr, double **sbufsouth_ptr,
                double **sbufeast_ptr, double **sbufwest_ptr,
                double **rbufnorth_ptr, double **rbufsouth_ptr,
                double **rbufeast_ptr, double **rbufwest_ptr)
{
    double *aold, *anew;
    double *sbufnorth, *sbufsouth, *sbufeast, *sbufwest;
    double *rbufnorth, *rbufsouth, *rbufeast, *rbufwest;

    /* allocate two working arrays */
    anew = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */
    aold = (double *) malloc((bx + 2) * (by + 2) * sizeof(double));     /* 1-wide halo zones! */

    memset(aold, 0, (bx + 2) * (by + 2) * sizeof(double));
    memset(anew, 0, (bx + 2) * (by + 2) * sizeof(double));

    /* allocate communication buffers */
    sbufnorth = (double *) malloc(bx * sizeof(double)); /* send buffers */
    sbufsouth = (double *) malloc(bx * sizeof(double));
    sbufeast = (double *) malloc(by * sizeof(double));
    sbufwest = (double *) malloc(by * sizeof(double));
    rbufnorth = (double *) malloc(bx * sizeof(double)); /* receive buffers */
    rbufsouth = (double *) malloc(bx * sizeof(double));
    rbufeast = (double *) malloc(by * sizeof(double));
    rbufwest = (double *) malloc(by * sizeof(double));

    memset(sbufnorth, 0, bx * sizeof(double));
    memset(sbufsouth, 0, bx * sizeof(double));
    memset(sbufeast, 0, by * sizeof(double));
    memset(sbufwest, 0, by * sizeof(double));
    memset(rbufnorth, 0, bx * sizeof(double));
    memset(rbufsouth, 0, bx * sizeof(double));
    memset(rbufeast, 0, by * sizeof(double));
    memset(rbufwest, 0, by * sizeof(double));

    (*aold_ptr) = aold;
    (*anew_ptr) = anew;
    (*sbufnorth_ptr) = sbufnorth;
    (*sbufsouth_ptr) = sbufsouth;
    (*sbufeast_ptr) = sbufeast;
    (*sbufwest_ptr) = sbufwest;
    (*rbufnorth_ptr) = rbufnorth;
    (*rbufsouth_ptr) = rbufsouth;
    (*rbufeast_ptr) = rbufeast;
    (*rbufwest_ptr) = rbufwest;
}

void alloc_dev_bufs(int bx, int by, double **aold_ptr, double **anew_ptr,
                    double **sbufnorth_ptr, double **sbufsouth_ptr,
                    double **sbufeast_ptr, double **sbufwest_ptr,
                    double **rbufnorth_ptr, double **rbufsouth_ptr,
                    double **rbufeast_ptr, double **rbufwest_ptr)
{
    double *aold, *anew;
    double *sbufnorth, *sbufsouth, *sbufeast, *sbufwest;
    double *rbufnorth, *rbufsouth, *rbufeast, *rbufwest;

    /* allocate two working arrays */
    cudaMalloc(&anew, ((bx + 2) * (by + 2)) * sizeof(double));  /* 1-wide halo zones! */
    cudaMalloc(&aold, ((bx + 2) * (by + 2)) * sizeof(double));  /* 1-wide halo zones! */

    cudaMemset(anew, 0, ((bx + 2) * (by + 2)) * sizeof(double));
    cudaMemset(aold, 0, ((bx + 2) * (by + 2)) * sizeof(double));

    /* allocate communication buffers */
    cudaMalloc(&sbufnorth, bx * sizeof(double));
    cudaMalloc(&sbufsouth, bx * sizeof(double));
    cudaMalloc(&sbufeast, by * sizeof(double));
    cudaMalloc(&sbufwest, by * sizeof(double));
    cudaMalloc(&rbufnorth, bx * sizeof(double));
    cudaMalloc(&rbufsouth, bx * sizeof(double));
    cudaMalloc(&rbufeast, by * sizeof(double));
    cudaMalloc(&rbufwest, by * sizeof(double));

    cudaMemset(sbufnorth, 0, bx * sizeof(double));
    cudaMemset(sbufsouth, 0, bx * sizeof(double));
    cudaMemset(sbufeast, 0, by * sizeof(double));
    cudaMemset(sbufwest, 0, by * sizeof(double));
    cudaMemset(rbufnorth, 0, bx * sizeof(double));
    cudaMemset(rbufsouth, 0, bx * sizeof(double));
    cudaMemset(rbufeast, 0, by * sizeof(double));
    cudaMemset(rbufwest, 0, by * sizeof(double));

    (*aold_ptr) = aold;
    (*anew_ptr) = anew;
    (*sbufnorth_ptr) = sbufnorth;
    (*sbufsouth_ptr) = sbufsouth;
    (*sbufeast_ptr) = sbufeast;
    (*sbufwest_ptr) = sbufwest;
    (*rbufnorth_ptr) = rbufnorth;
    (*rbufsouth_ptr) = rbufsouth;
    (*rbufeast_ptr) = rbufeast;
    (*rbufwest_ptr) = rbufwest;
}

void free_bufs(double *aold, double *anew,
               double *sbufnorth, double *sbufsouth,
               double *sbufeast, double *sbufwest,
               double *rbufnorth, double *rbufsouth, double *rbufeast, double *rbufwest)
{
    free(aold);
    free(anew);
    free(sbufnorth);
    free(sbufsouth);
    free(sbufeast);
    free(sbufwest);
    free(rbufnorth);
    free(rbufsouth);
    free(rbufeast);
    free(rbufwest);
}

void free_dev_bufs(double *aold, double *anew,
                   double *sbufnorth, double *sbufsouth,
                   double *sbufeast, double *sbufwest,
                   double *rbufnorth, double *rbufsouth, double *rbufeast, double *rbufwest)
{
    cudaFree(aold);
    cudaFree(anew);
    cudaFree(sbufnorth);
    cudaFree(sbufsouth);
    cudaFree(sbufeast);
    cudaFree(sbufwest);
    cudaFree(rbufnorth);
    cudaFree(rbufsouth);
    cudaFree(rbufeast);
    cudaFree(rbufwest);
}

__global__ void update_sources(int bx, int by, double heat, int nsources, int *locsources,
                               double *aold)
{
    int (*locsources_)[2] = (int (*)[2]) locsources;
    int i = 1 + (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = 1 + (blockIdx.y * blockDim.y) + threadIdx.y;

    /* A kernel for updating a few doubles in the grid is overkill
     * and ideally should be done during update of the grid. However,
     * for sake of demonstration this should be fine. */
    if (i < (bx + 1) && j < (by + 1)) {
        for (int k = 0; k < nsources; k++) {
            if (i == locsources_[k][0] && j == locsources_[k][1])
                aold[ind(i, j)] += heat;
        }
    }
}

__global__ void pack_data(int bx, int by, double *aold,
                          double *sbufnorth, double *sbufsouth, double *sbufeast, double *sbufwest)
{
    /* Get thread idx in global domain */
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    /* First do the north buf packing */
    if (i < bx)
        sbufnorth[i] = aold[ind(i + 1, 1)];

    /* Next do the south buf packing */
    if (i < bx)
        sbufsouth[i] = aold[ind(i + 1, by)];

    /* Then do the east buf packing */
    if (i < by)
        sbufeast[i] = aold[ind(bx, i + 1)];

    /* Finally do the west buf packing */
    if (i < by)
        sbufwest[i] = aold[ind(1, i + 1)];
}

__global__ void unpack_data(int bx, int by, double *aold,
                            double *rbufnorth, double *rbufsouth, double *rbufeast,
                            double *rbufwest)
{
    /* Get thread idx in global domain */
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    /* First do the north buf unpacking */
    if (i < bx)
        aold[ind(i + 1, 0)] = rbufnorth[i];

    /* Next do the south buf unpacking */
    if (i < bx)
        aold[ind(i + 1, by + 1)] = rbufsouth[i];

    /* Then do the east buf unpacking */
    if (i < by)
        aold[ind(bx + 1, i + 1)] = rbufeast[i];

    /* Finally do the west buf unpacking */
    if (i < by)
        aold[ind(0, i + 1)] = rbufwest[i];
}

__global__ void update_grid(int bx, int by, double *aold, double *anew, double *heat)
{
    /* Shared memory size specified at kernel launch */
    extern __shared__ double heat_[];
    int blkDimX, blkDimY;

    /* Calculate index in matrix for thread in threadblock */
    int i = 1 + (blockDim.x * blockIdx.x) + threadIdx.x;
    int j = 1 + (blockDim.y * blockIdx.y) + threadIdx.y;
    int ii = threadIdx.x;
    int jj = threadIdx.y;
    blkDimX = MIN(blockDim.x, bx - (blockDim.x * blockIdx.x));

    /* Update heat value at thread location */
    if (i < (bx + 1) && j < (by + 1)) {
        anew[ind(i, j)] =
            anew[ind(i, j)] / 2.0 + (aold[ind(i - 1, j)] +
                                     aold[ind(i + 1, j)] +
                                     aold[ind(i, j - 1)] + aold[ind(i, j + 1)]) / 4.0 / 2.0;

        /* store partial heat values in shared memory */
        heat_[(jj * blkDimX) + ii] = anew[ind(i, j)];
    }

    /* Wait for all threads in the block to complete */
    __syncthreads();

    /* First thread in threadblock reduces heat values into global memory:
     * could be done more efficiently with a hierarchical reduction; Again,
     * here we don't care about performance. */
    if (ii == 0 && jj == 0) {
        /* account for cases in which the thread block is not multiple of
         * bx and/or by */
        blkDimX = MIN(blockDim.x, bx - (blockDim.x * blockIdx.x));
        blkDimY = MIN(blockDim.y, by - (blockDim.y * blockIdx.y));
        double reduce = 0.0;
        for (j = 0; j < blkDimY; j++)
            for (i = 0; i < blkDimX; i++)
                reduce += heat_[(j * blkDimX) + i];
        heat[(blockIdx.y * gridDim.x) + blockIdx.x] = reduce;
    }
}
