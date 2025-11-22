#!/bin/bash
#SBATCH --account=training2557
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=dc-gpu-devel
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
#SBATCH --time=02:00:00

echo "=== Job Started at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Change to working directory
cd /p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples
echo "Working directory: $(pwd)"

# Load modules
echo "=== Loading Modules ==="
module load GCC
module load CUDA
module load PyTorch
module load torchvision

# Check GPU
echo "=== GPU Check ==="
nvidia-smi

# Run ensemble attack script (unbuffered for real-time output)
echo "=== Starting Attack Script ==="
python -u run_pgd_ensemble.py

echo "=== Job Finished at $(date) ==="

