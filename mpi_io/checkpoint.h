/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *
 *  (C) 2004 by University of Chicago.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef CHECKPOINT_H_
#define CHECKPOINT_H_

int STENCILIO_Init(MPI_Comm comm);
int STENCILIO_Can_restart(void);
int STENCILIO_Info_set(MPI_Info info);
int STENCILIO_Restart(char *prefix, double *matrix, int n,
                      int *coords, int px, int py, int iter,
                      MPI_Info info);
int STENCILIO_Checkpoint(char *prefix, double *matrix, int n,
                         int *coords, int px, int py, int iter,
                         MPI_Info info);
int STENCILIO_Finalize(void);

#endif /* CHECKPOINT_H_ */
