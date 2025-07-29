/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include <stdio.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
    int rank;
    int size;
    MPI_Session session;
    MPI_Group group;
    MPI_Comm comm;

    MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_ABORT, &session);
    MPI_Group_from_session_pset(session, "mpi://WORLD", &group);
    MPI_Comm_create_from_group(group, "hello_session", MPI_INFO_NULL, MPI_ERRORS_ABORT, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    printf("Hello from process %d of %d\n", rank, size);
    MPI_Group_free(&group);
    MPI_Comm_free(&comm);
    MPI_Session_finalize(&session);
    return 0;
}
