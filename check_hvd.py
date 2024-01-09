#To check support of some libraries e.g. MPI, NCCL
#see here: https://horovod.readthedocs.io/en/stable/api.html

import horovod.tensorflow as hvd

hvd.init()

if hvd.rocm_built():
    print("--Horovod is compiled with ROCm support.")
else:
    print("--Horovod is NOT compiled with ROCm support.")

if hvd.cuda_built():
    print("--Horovod is compiled with CUDA support.")
else:
    print("--Horovod is NOT compiled with CUDA support.")

if hvd.mpi_enabled():
    print("--MPI support is enabled.")
else:
    print("--MPI support is not enabled.")

if hvd.mpi_built():
    print("--MPI is compiled with horovod support.")
else:
    print("--MPI is NOT compiled with horovod support.")

if hvd.nccl_built():
    print("--Horovod is compiled with NCCL support.")
else:
    print("--Horovod is NOT compiled with NCCL support.")

if hvd.mpi_threads_supported():
    print("--MPI multi-threading is supported.")
    print("You may mix and match Horovod usage with other MPI libraries, such as mpi4py.")
else:
    print("--MPI multi-threading is NOT supported.")

print('Hello, rank = %d, local_rank = %d, size = %d, local_size = %d' % (hvd.rank(), hvd.local_rank(), hvd.size(), hvd.local_size()))
