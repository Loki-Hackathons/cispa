#!/bin/bash
#SBATCH --account=training2557
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=output/%j.log
#SBATCH --error=output/%j.log

# Load modules
module load GCC CUDA PyTorch torchvision

# Move to project root (task_1_adversarial_examples)
# We assume this script is at .../task_1_adversarial_examples/version2/run_solver.sh
# So we go up one level
cd /p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples

# Activate Venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run Solver as a module to handle imports correctly
echo "Starting Solver..."
python -u -m version2.main_solver --batch-size 20 --steps 100 --epsilon 12.0

echo "Solver Finished."
