#!/bin/bash

export TAU_MAKEFILE=/home/flyxian/.local/tau/x86_64/lib/Makefile.tau-papi-mpi-pdt

tau_cc.sh -tau_options="-optCompInst -optVerbose" -o printarr_par.o -c ../common/printarr_par.c
tau_cc.sh -tau_options="-optCompInst -optVerbose" -o stencil_hotspot.o -c stencil_hotspot.c
tau_cc.sh -tau_options="-optCompInst -optVerbose" -o stencil_hotspot printarr_par.o stencil_hotspot.o

#cat > select.tau << EOF
#BEGIN_INSTRUMENT_SECTION
#loops routine=“#”
#END_INSTRUMENT_SECTION
#EOF

#tau_cc.sh -tau_options="-optTauSelectFile=select.tau -optRevert -optVerbose" -o printarr_par.o -c printarr_par.c
#tau_cc.sh -tau_options="-optTauSelectFile=select.tau -optRevert -optVerbose" -o stencil_hotspot.o -c stencil_hotspot.c
#tau_cc.sh -tau_options="-optTauSelectFile=select.tau -optRevert -optVerbose" -o stencil_hotspot printarr_par.o stencil_hotspot.o

