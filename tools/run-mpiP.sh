#!/bin/bash

export MPIR_CVAR_NOLOCAL=1

mpiexec -env MPIP="-t 10 -k 2 -c" -n 16 ./stencil_hotspot 1024 100 10 4 4

