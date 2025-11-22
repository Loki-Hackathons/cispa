#!/bin/bash
#SBATCH --account=training2557
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu-devel
#SBATCH --output=output/%j.out
#SBATCH --time=02:00:00

# Load necessary modules
module load GCC
module load CUDA
module load PyTorch
module load torchvision

# Navigate to task directory
cd /p/home/jusers/dougnon1/jureca/code/cispa-hackathon/task_2_image_attribution

# Verify GPU allocation
echo "=========================================="
echo "GPU Allocation:"
nvidia-smi --list-gpus
echo "=========================================="
echo ""

# Run training
echo "Starting training with 4x A100 GPUs..."
python3 main_attribution.py 2>&1 | tee training_${SLURM_JOB_ID}.log

# Submit results if submission.csv exists
if [ -f "submission.csv" ]; then
    echo ""
    echo "=========================================="
    echo "Submitting results..."
    echo "=========================================="
    python3 submit.py
else
    echo "Warning: submission.csv not found!"
fi

echo ""
echo "Job completed!"

