/* -*- mode: c; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * see copyright in top-level directory.
 */

#ifndef STENCIL_PAR_H_INCLUDED
#define STENCIL_PAR_H_INCLUDED

#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define NSOURCES (3)

void printarr_par(int iter, double *array, int size, int px, int py, int rx, int ry, int bx, int by,
                  int offx, int offy, int (*ind)(int i, int j, int bx), MPI_Comm comm);
void setup(int rank, int proc, int argc, char **argv,
           int *n_ptr, int *energy_ptr, int *niters_ptr, int *px_ptr, int *py_ptr, int *final_flag);

void init_sources(int bx, int by, int offx, int offy, int n,
                  const int nsources, int sources[][2], int *locnsources_ptr, int locsources[][2]);

void refresh_heat_source(int bx, int nsources, int sources[][2], int energy, double *aold_ptr);

void alloc_bufs(int bx, int by, double **aold_ptr, double **anew_ptr);

#ifndef __NVCC__
void update_grid(int bx, int by, double *aold, double *anew, double *heat_ptr);
#endif

void free_bufs(double *aold, double *anew);

#endif /* STENCIL_PAR_H_INCLUDED */
