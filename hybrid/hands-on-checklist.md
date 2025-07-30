## Hybrid MPI Programming Hands-on Checklist

* [ ] Basic send/recv
    * [ ] Add timing using MPI_Wtime
    * [ ] Aggregate and average to improve precision
    * [ ] Calculate latency - ____ μs
    * [ ] Calculate bandwidth - ____ GB/sec
    * [ ] Plot latency vs message size
    * [ ] Plot bandwidth vs message size
    * Bonuses
        * [ ] Multiple measurements and observe variations
        * [ ] Reduce variations by warm-up
        * [ ] Quantify variations by measure uncertainties
        * [ ] Measure inter-node (send/recv between two nodes)
        * [ ] Measure intra-node (send/recv between processes on the same node)
        * [ ] Improve results by binding processes
            * [ ] Measure inter-NUMA
            * [ ] Measure intra-NUMA

* MPI+Threads
    * [ ] OpenMP parallel send/recv
    * [ ] Measure message rate (messages/sec) vs number of threads
    * [ ] Measure aggregate bandwidth vs number of threads
    * Bonuses
        * [ ] Vary message sizes
        * [ ] Use separate communicator per thread
        * [ ] Enable multiple VCIs
        * [ ] Stencil example using MPI+Threads

* MPI+GPU
   * [ ] Allocate host buffer, device buffer, registered host buffer, and shared buffer
   * [ ] Send/recv between various types of buffers
   * [ ] Inter-node, intra-node, inter-device, inter-tiles
   * Bonuses
       * [ ] Stencil example using MPI+GPU

References:
* [Aurora guide](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/)
* [Intel OpenMP](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-2/openmp-runtime-library-routines.html)
* [MPICH manpages](https://www.mpich.org/static/docs/v4.3.0/)
* [mpiexec manpage](https://cpe.ext.hpe.com/docs/24.03/workload-manager/mpiexec.html)
* [Cray MPICH](https://cpe.ext.hpe.com/docs/24.03/mpt/mpich/intro_mpi.html)

