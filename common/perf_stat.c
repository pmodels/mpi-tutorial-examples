/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include "perf_stat.h"

#define MAX_NAME_LEN 16

static char timer_name[N_TIMERS][MAX_NAME_LEN] = {
    "Execution",
    "Computation",
    "Communication",
    "Imaging"
};

double t_total[N_TIMERS] = { 0.0 };

#ifdef MPI_INCLUDED
double t_start[N_TIMERS] = { 0.0 };
double t_stop[N_TIMERS] = { 0.0 };
#else
struct timespec t_start[N_TIMERS];
struct timespec t_stop[N_TIMERS];
#endif

void PERF_PRINT()
{
    for (int i = 0; i < N_TIMERS; i++) {
        printf("Total %16s time: %f s\n", timer_name[i], t_total[i]);
    }
}
