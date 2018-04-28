#!/bin/bash

export MPIR_CVAR_NOLOCAL=1

rm *.clog2 *.slog2

mpiexec -n 16 ./stencil_hotspot 1024 100 10 4 4

clog2TOslog2 ./stencil_hotspot.clog2
