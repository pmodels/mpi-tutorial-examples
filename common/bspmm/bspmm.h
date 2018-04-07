#ifndef BSPMM_H_
#define BSPMM_H_

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <time.h>

#define BLK_DIM 128
#define SPARSITY_A 0.8
#define SPARSITY_B 0.8
#define RAND_RANGE 10

int setup(int rank, int nprocs, int argc, char **argv, int *mat_dim_ptr);
void init_mats(int mat_dim, double *mem, double **mat_a_ptr, double **mat_b_ptr,
               double **mat_c_ptr);
void dgemm(double *local_a, double *local_b, double *local_c);
int is_zero_local(double *local_mat);
int is_zero_global(double *global_mat, int mat_dim, int global_i, int global_j);
void pack_global_to_local(double *local_mat, double *global_mat, int mat_dim, int global_i,
                          int global_j);
void unpack_local_to_global(double *global_mat, double *local_mat, int mat_dim, int global_i,
                            int global_j);
void add_local_to_global(double *global_mat, double *local_mat, int mat_dim, int global_i,
                         int global_j);
void check_mats(double *mat_a, double *mat_b, double *mat_c, int mat_dim);

#endif /* BSPMM_H_ */
