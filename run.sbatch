#!/bin/bash
#SBATCH --job-name=mace-mliap
#SBATCH --partition=beagle3
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=run-%j.out
#SBATCH --account=pi-gavoth         # workshop account
##SBATCH --reservation=[reservation]   # workshop reservation

# Usage:  sbatch run.sbatch                    (float64)
#         sbatch --export=ALL,DTYPE=float32 run.sbatch
set -e
cd "$SLURM_SUBMIT_DIR"

module load openmpi-cuda-aware/4.1.2+cuda-12.2 python/miniforge-24.1.2
MLIAP_ENV=/software/python-miniforge-24.1.2-el8-x86_64/envs/mliap-env   # torch, mace-torch, cuequivariance, lamm$
source activate $MLIAP_ENV
export PYTHONNOUSERSITE=1      # ignore packages in ~/.local
export OMP_NUM_THREADS=1

# LAMMPS with KOKKOS(CUDA) + ML-IAP + PYTHON; a Kokkos build targets one GPU architecture.
case "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" in
  *V100*) export LMP=$MLIAP_ENV/bin/lmp_v100 ;;
  *A100*) export LMP=$MLIAP_ENV/bin/lmp_a100 ;;
  *)      export LMP=none ;;
esac

DTYPE=${DTYPE:-float64}
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)   LAMMPS: $LMP   dtype: $DTYPE"

# ---- Step 1: convert the MACE model to an ML-IAP file -------------------------------------
# Done here on the GPU node on purpose: cuequivariance chooses its CUDA kernels at conversion
# time. Converting on a login node (no GPU) gives a file that runs ~25x slower.
ln -sf c-h2o.model c-h2o-$DTYPE.model                # output name = <input>-mliap_lammps.pt
python -m mace.cli.create_lammps_model c-h2o-$DTYPE.model --format mliap --dtype $DTYPE
MODEL=c-h2o-$DTYPE.model-mliap_lammps.pt

# ---- Step 2: run 100 MD steps ---------------------------------------------------------------
mpirun -np 1 $LMP -in in.water.mliap -v model $MODEL \
    -k on g 1 -sf kk -pk kokkos neigh half newton on \
    -log log.$DTYPE.lammps
