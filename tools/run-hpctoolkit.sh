#!/bin/bash

export MPIR_CVAR_NOLOCAL=1

mpiexec -n 16 hpcrun ./stencil_hotspot 1024 100 10 4 4

