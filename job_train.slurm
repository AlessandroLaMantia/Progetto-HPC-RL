#!/bin/bash
#SBATCH --job-name=quantum-ppo-ghz
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --ntasks-per-node=6


module load miniconda3
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate quantum-rl
cd "$SLURM_SUBMIT_DIR"
python 3.run_repeated_training.py --target custom --target-file targets/random_target_3_qubit_normalized.json --qubits 3 --max-gates 30 --iterations 10 --workers 2 --episodes 10 --runs 10 --checkpoint-root checkpoints_random3 --output training_random3.csv
conda deactivate
