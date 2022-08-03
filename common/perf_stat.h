/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef PERF_STAT_H
#define PERF_STAT_H

typedef enum {
    TIMER_EXEC = 0,
    TIMER_COMP = 1,
    TIMER_COMM,
    TIMER_IMG,
    N_TIMERS
} perf_timer_labels;

extern double t_total[N_TIMERS];

#ifdef MPI_INCLUDED

extern double t_start[N_TIMERS];
extern double t_stop[N_TIMERS];

#define PERF_TIMER_BEGIN(timer) \
    do { \
        t_start[(timer)] = MPI_Wtime(); \
    } while (0)

#define PERF_TIMER_END(timer) \
    do { \
        t_stop[(timer)] = MPI_Wtime(); \
        t_total[(timer)] += t_stop[(timer)] - t_start[(timer)]; \
    } while (0)

#else

#include <time.h>

extern struct timespec t_start[N_TIMERS];
extern struct timespec t_stop[N_TIMERS];

#define TIME_STRUCT_DIFF(end, start) \
    ((double)((end).tv_sec - (start).tv_sec) \
     + 1.0e-9 * (double)((end).tv_nsec - (start).tv_nsec))

#define PERF_TIMER_BEGIN(timer) \
    do { \
        clock_gettime(CLOCK_REALTIME, &t_start[(timer)]); \
    } while (0)

#define PERF_TIMER_END(timer) \
    do { \
        clock_gettime(CLOCK_REALTIME, &t_stop[(timer)]); \
        t_total[(timer)] += TIME_STRUCT_DIFF(t_stop[(timer)], t_start[(timer)]); \
    } while (0)

#endif

void PERF_PRINT();

#endif /* PERF_STAT_H */
