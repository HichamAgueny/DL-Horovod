# Distributed Deep Learning with Horovod
This course is part of the [NLDL2024](https://www.nldl.org/program/winter-school) winter school at UiT - The Arctic University of Norway. It is about distributed deep learning with Horovod. 

## Setup Horovod-TensorFlow-rocm
Here is a step-by-step guide on installing Horovod-TensorFlow-rocm:

### In a virtual environment

- Load modules:
```
$ module load LUMI/22.08 partition/G
$ module load cray-python/3.9.12.1
```
- Create a virtual environment:
```
python -m venv HvdTF2.10.1_rocm5.2.3_python3.9.12.1/
source HvdTF2.10.1_rocm5.2.3_python3.9.12.1/bin/activate
```
- Install TensorFlow-rocm:
```
python -m pip install tensorflow-rocm==2.10.1
```
- Install additional packages:
```
python -m pip install -r req.txt
```

**N.B** The package `tensorflow-probability==0.21` is not compatible with `tensorflow==2.10.1`. It requires a lower version e.g. `tensorflow-probability==0.19.0`.

- Install Horovod:
  ```bash
  export HOROVOD_WITHOUT_MXNET=1
  export HOROVOD_WITHOUT_PYTORCH=1
  export HOROVOD_GPU=ROCM
  export HOROVOD_GPU_OPERATIONS=NCCL
  export HOROVOD_WITHOUT_GLOO=1
  export HOROVOD_WITH_TENSORFLOW=1
  export HOROVOD_ROCM_PATH=/opt/rocm
  export HOROVOD_RCCL_HOME=/opt/rocm/rccl
  export RCCL_INCLUDE_DIRS=/opt/rocm/rccl/include
  export HOROVOD_RCCL_LIB=/opt/rocm/rccl/lib
  export HOROVOD_MPICXX_SHOW="CC --cray-print-opts=all"
  export HCC_AMDGPU_TARGET=gfx90a
 
  CXX=CC pip install --no-cache-dir --force-reinstall horovod[tensorflow-rocm,keras]==0.28.1
  ```

- Check the built of Horovod:
  
To check if Horovod is built properly with TensorFlow, MPI and NCCL, run this command
```
horovodrun --check-build
```
Running the command above should display the following:
```bash
Available Frameworks:
    [X] TensorFlow
    [ ] PyTorch
    [ ] MXNet

Available Controllers:
    [X] MPI
    [ ] Gloo

Available Tensor Operations:
    [X] NCCL
    [ ] DDL
    [ ] CCL
    [X] MPI
    [ ] Gloo 
 ```   

### How to build `aws-ofi-rccl` with EasyBuild:

To build `rccl` (a library for communication between GPUs) and [`aws-ofi-rccl`](https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl) plugin with [EasyBuild](https://docs.lumi-supercomputer.eu/software/installing/easybuild/), follow this step-by-step guide:

**Step 0:** Specify the path for EasyBuild installation**
```
$ export EBU_USER_PREFIX=/project/project_46xxxxxx/EasyBuild
```
**Step 1:** Load the LUMI software stack
```
$ module load LUMI/22.08 partition/G
$ module load rocm/5.2.3
```

**Step 2:** Load EasyBuild
```
$ module load EasyBuild-user
```

**Step 3:** Install `aws-ofi-rccl`
```
$ eb aws-ofi-rccl-66b3b31-cpeGNU-22.08.eb -r
```

**When loading `module load aws-ofi-rccl` both `rccl/2.12.7-cpeGNU-22.08` and `aws-ofi-rccl/66b3b31-cpeGNU-22.08` will be loaded.**

**N.B.:** To enable Peer-2-Peer communication via PCIe-connected GPUs: `export HSA_FORCE_FINE_GRAIN_PCIE=1`

## How to run Horovod-TensorFlow-based application

Here is a SLURM job example for running a Horovod-TensorFlow-based application

```bash
#!/bin/bash -e
#SBATCH --job-name=8gpu_hvd
#SBATCH --account=project_465xxxxxx
#SBATCH --time=00:30:00
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=8
#SBATCH --gpus-per-node=8
#SBATCH -o %x-%j.out
#SBATCH --exclusive
##SBATCH --mem-per-gpu=60G

N=$SLURM_JOB_NUM_NODES
Ntasks=$SLURM_NTASKS
Ntasks_per_node=$((Ntasks / N))
#$SLURM_TASKS_PER_NODE

echo "--nbr of nodes:" $N
echo "--total nbr of gpus:" $Ntasks
echo "--nbr of gpus per nodes :" $Ntasks_per_node

# Set the environment variable EBU_USER_PREFIX
export EBU_USER_PREFIX=/project/project_465xxxxxx/EasyBuild

ml LUMI/22.08  partition/G
ml rocm/5.2.3
ml cray-python/3.9.12.1
ml craype-accel-amd-gfx90a
ml aws-ofi-rccl

ml

# Uncomment to see RCCL init and collectives info
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,COLL
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1

#
workdir=/project/project_465xxxxxxx
#Path to the application
SRC=$workdir/DL_Horovod/TF_Horovod/main_hvd.py

#My virtual env. for installing additional packages
MyVirtEnv=${workdir}/HvdTF2.10.1_rocm5.2.3_python3.9.12.1

# Binding each CPU-core to the closeset GPU
MYMASKS="0x${fe}000000000000,0x${fe}00000000000000,0x${fe}0000,0x${fe}000000,0x${fe},0x${fe}00,0x${fe}00000000,0x${fe}0000000000"

# Collect the master address as that has to be calculated out of the container.
#
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)

source ${MyVirtEnv}/bin/activate

launch_syntax="python -u ${SRC}"

time srun --cpu-bind=mask_cpu:$MYMASKS \
     --nodes $N --gpus $Ntasks --gpus-per-node $Ntasks_per_node --ntasks-per-node $Ntasks_per_node \
     python -u ${SRC}
 ```    
