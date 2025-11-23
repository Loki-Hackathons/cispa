#!/bin/bash
#SBATCH --account=training2557
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --job-name=lock_ram_v3

# LOCK & RAM Strategy V3: Ultra-Aggressive
# - Uses API results to identify failures (not local surrogate)
# - Freezes SUCCESS images from API
# - ULTRA-AGGRESSIVE attack on remaining FAILED images
# - Parallelizes across 2 GPUs
# - Ultra-aggressive parameters: kappa=50, epsilon=15, steps=300, restarts=15

echo "=================================================================="
echo "LOCK & RAM Strategy V3: Ultra-Aggressive (epsilon=15)"
echo "=================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "GPUs: 2 (parallelized)"
echo "Expected duration: ~40-60 minutes for ~18 failed images"
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

# Check if previous submission exists
PREVIOUS_SUB="output/submission_ram_v2.npz"
if [ ! -f "$PREVIOUS_SUB" ]; then
    echo "ERROR: Previous submission not found: $PREVIOUS_SUB"
    echo "Please run lock_and_ram_v2.py first to generate submission_ram_v2.npz"
    exit 1
fi

echo "✓ Found previous submission: $PREVIOUS_SUB"

# Find API logits JSON (REQUIRED - uses API results, not local)
API_LOGITS_JSON=""
if [ -n "$(ls logs/api/logits_submission_ram_v2_*.json 2>/dev/null)" ]; then
    API_LOGITS_JSON=$(ls -t logs/api/logits_submission_ram_v2_*.json 2>/dev/null | head -1)
    echo "✓ Found API logits JSON: $API_LOGITS_JSON"
else
    echo "❌ ERROR: API logits JSON not found!"
    echo "  Required: logs/api/logits_submission_ram_v2_*.json"
    echo "  Generate it with: python analyze.py output/submission_ram_v2.npz --mode api"
    exit 1
fi

# Run Lock & Ram V3 strategy
# Parameters (ULTRA-AGGRESSIVE):
# - epsilon: 30.0 (fixed, no binary search) - allows norm ~0.62 (vs 0.2 before)
# - kappa: 50.0 (huge margin requirement)
# - pgd_steps: 250 (same as V2)
# - restarts: 10 (same as V2)
# - Input Diversity: ENABLED (default)
# - Multi-GPU: 2 GPUs parallelized

python -u lock_and_ram_v2.py \
    --previous-submission "$PREVIOUS_SUB" \
    --api-logits-json "$API_LOGITS_JSON" \
    --dataset ../natural_images.pt \
    --output-dir ./output \
    --log-dir ./logs \
    --output-name submission_ram_v3.npz \
    --epsilon 30.0 \
    --kappa 50.0 \
    --pgd-steps 250 \
    --restarts 10 \
    --alpha-factor 2.5 \
    --momentum 0.9 \
    --num-gpus 2

exit_code=$?

echo ""
echo "=================================================================="
echo "Job completed: $(date)"
echo "Exit code: $exit_code"
echo "=================================================================="

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ Lock & Ram V3 strategy completed successfully!"
    echo ""
    echo "=================================================================="
    echo "SUBMISSION INSTRUCTIONS"
    echo "=================================================================="
    echo ""
    echo "1. Review results locally (fast):"
    echo "   python analyze.py output/submission_ram_v3.npz --mode local"
    echo ""
    echo "2. Test transfer with API (wait 15 min after last query if needed):"
    echo "   python analyze.py output/submission_ram_v3.npz --mode api"
    echo ""
    echo "3. Submit to leaderboard (wait 5 min after last submission if needed):"
    echo "   python submit.py output/submission_ram_v3.npz --action submit"
    echo ""
    echo "=================================================================="
    echo "NOTE: API rate limits:"
    echo "  - Logits query: 1 every 15 minutes"
    echo "  - Submission: 1 every 5 minutes"
    echo "=================================================================="
else
    echo ""
    echo "❌ Lock & Ram V3 strategy failed with exit code $exit_code"
    echo "Check logs/slurm_${SLURM_JOB_ID}.err for details"
fi

exit $exit_code

