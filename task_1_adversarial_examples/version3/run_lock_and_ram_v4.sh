#!/bin/bash
#SBATCH --account=training2557
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples/version3/logs/slurm_%j.out
#SBATCH --error=/p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples/version3/logs/slurm_%j.err
#SBATCH --job-name=lock_ram_v4
#SBATCH --chdir=/p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples/version3

# LOCK & RAM Strategy V4: Adaptive Epsilon per Image
# - Uses API results to identify failures (not local surrogate)
# - Adapts epsilon per image based on previous L2 distances
# - Iteratively refines epsilon to minimize L2 distance
# - Supports multiple API iterations with intelligent epsilon adjustment
# - Parallelizes across 2 GPUs
# - Parameters: kappa=50, steps=80, restarts=5

echo "=================================================================="
echo "LOCK & RAM Strategy V4: Adaptive Epsilon per Image"
echo "=================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "GPUs: 2 (parallelized)"
echo "Expected duration: ~20-30 minutes"
echo "=================================================================="

# Load modules
module load GCC CUDA PyTorch torchvision

# Navigate to version3 directory
cd /p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples/version3

# Check GPUs
echo ""
echo "=== GPU Information ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo ""

# Check if previous submission exists (iteration 3)
PREVIOUS_SUB="output/submission_ram_v4.npz"
if [ ! -f "$PREVIOUS_SUB" ]; then
    echo "ERROR: Previous submission not found: $PREVIOUS_SUB"
    echo "Please run lock_and_ram_v4.py (iteration 3) first to generate submission_ram_v4.npz"
    exit 1
fi

echo "✓ Found previous submission (iteration 3): $PREVIOUS_SUB"

# Find analysis JSON (REQUIRED - contains L2 distances)
ANALYSIS_JSON=""
if [ -n "$(ls logs/analysis_api_*.json 2>/dev/null)" ]; then
    ANALYSIS_JSON=$(ls -t logs/analysis_api_*.json 2>/dev/null | head -1)
    echo "✓ Found analysis JSON: $ANALYSIS_JSON"
else
    echo "❌ ERROR: Analysis JSON not found!"
    echo "  Required: logs/analysis_api_*.json"
    echo "  Generate it with: python analyze.py output/submission_ram_v3.npz --mode api"
    exit 1
fi

# Run Lock & Ram V4 strategy
# Parameters:
# - epsilon: Adaptive per image (based on previous L2 distances)
# - kappa: 50.0 (huge margin requirement)
# - pgd_steps: 80 (reduced for speed)
# - restarts: 5 (reduced for speed)
# - Input Diversity: ENABLED (default)
# - Multi-GPU: 2 GPUs parallelized
# - Attacks ALL images (not just FAILED) for L2 optimization

python -u lock_and_ram_v4.py \
    --previous-submission "$PREVIOUS_SUB" \
    --analysis-json "$ANALYSIS_JSON" \
    --dataset ../natural_images.pt \
    --output-dir ./output \
    --log-dir ./logs \
    --output-name submission_ram_v4_final.npz \
    --kappa 50.0 \
    --pgd-steps 60 \
    --restarts 3 \
    --num-gpus 2 \
    --final-iteration

exit_code=$?

echo ""
echo "=================================================================="
echo "Job completed: $(date)"
echo "Exit code: $exit_code"
echo "=================================================================="

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ Lock & Ram V4 strategy completed successfully!"
    echo ""
    echo "=================================================================="
    echo "NEXT STEPS FOR ITERATION"
    echo "=================================================================="
    echo ""
    echo "1. Analyze new submission with API:"
    echo "   python analyze.py output/submission_ram_v4.npz --mode api"
    echo ""
    echo "2. Review epsilon mapping:"
    echo "   cat logs/lock_and_ram_v4_epsilon_mapping.json"
    echo ""
    echo "3. For next iteration, update run_lock_and_ram_v4.sh:"
    echo "   - Set PREVIOUS_SUB to output/submission_ram_v4.npz"
    echo "   - The script will auto-detect the latest analysis JSON"
    echo "   - Run: sbatch run_lock_and_ram_v4.sh"
    echo ""
    echo "4. The system will automatically adjust epsilons based on results"
    echo ""
    echo "=================================================================="
else
    echo ""
    echo "❌ Lock & Ram V4 strategy failed with exit code $exit_code"
    echo "Check logs/slurm_${SLURM_JOB_ID}.err for details"
fi

exit $exit_code

