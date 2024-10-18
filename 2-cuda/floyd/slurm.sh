#!/bin/bash -l
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --export=ALL
#SBATCH --output=slurm.out

# Load the CUDA compiler.
module load system/CUDA

# Load the Cmake
module load devel/CMake
module load devel/Doxygen

#compile code
cmake -B build -S .
cmake --build build

./build/floyd 100 64 64
