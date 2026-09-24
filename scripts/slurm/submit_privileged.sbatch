#!/usr/bin/env bash
#SBATCH --job-name=priv-opd
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output=logs/slurm-%x-%j.out
# Fill in for your cluster:
##SBATCH --partition=<partition>
##SBATCH --account=<account>
##SBATCH --time=<hh:mm:ss>
#
# One independent run per job; --exclusive because the run starts and force-stops its own Ray head.
#   sbatch --export=ALL,DATASET=openthoughts,METHOD=ours_solution scripts/slurm/submit_privileged.sbatch
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
exec bash scripts/run_privileged_experiment.sh "${DATASET:?set DATASET}" "${METHOD:?set METHOD}"
