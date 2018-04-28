#!/bin/bash

export MPIR_CVAR_NOLOCAL=1

# mpiexec -n 16 tau_exec ./stencil_hotspot 1024 100 10 4 4

if test -d "$PWD/profile-tau"; then
    rm -rf "$PWD/profile-tau"
fi

# export PROFILEDIR=$PWD/profile-tau
# export TAU_COMM_MATRIX=1
export TAU_CALLPATH=1
export TAU_CALLPATH_DEPTH=100
export TAU_METRICS=TIME:PAPI_FP_INS:PAPI_L1_DCM
mpiexec -n 16 ./stencil_hotspot 1024 100 10 4 4

# if test -d "$PWD/trace-tau"; then
#     rm -rf "$PWD/trace-tau"
#     mkdir "$PWD/trace-tau"
# else
#     mkdir "$PWD/trace-tau"
# fi

# export TAU_VERBOSE=1
# export TAU_TRACE=1
# export TRACEDIR=$PWD/trace-tau
# mpiexec -n 16 ./stencil_hotspot 1024 100 10 4 4

