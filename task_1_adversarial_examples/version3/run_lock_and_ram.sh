#!/bin/bash
#SBATCH --account=training2557
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --job-name=lock_and_ram

# LOCK & RAM Strategy: Freeze & Boost
# - Freezes 32 SUCCESS images from submission_fast.npz
# - Aggressively attacks only 68 FAILED images
# - Ultra-fast: No Binary Search, fixed epsilon, high kappa
# - Expected duration: ~10-15 minutes

echo "=================================================================="
echo "LOCK & RAM Strategy: Freeze & Boost"
echo "=================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Expected duration: ~10-15 minutes (68 images only, no BS)"
echo "=================================================================="

# Load modules
module load GCC CUDA PyTorch torchvision

# Navigate to version3 directory
cd /p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples/version3

# Check GPU
echo ""
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Check if previous submission exists
PREVIOUS_SUB="output/submission_fast.npz"
if [ ! -f "$PREVIOUS_SUB" ]; then
    echo "ERROR: Previous submission not found: $PREVIOUS_SUB"
    echo "Please run run_solver_FAST.sh first to generate submission_fast.npz"
    exit 1
fi

echo "✓ Found previous submission: $PREVIOUS_SUB"

# Check if logits JSON exists (try to find it)
LOGITS_JSON=""
if [ -f "logs/api/logits_submission_fast_20251123_024816.json" ]; then
    LOGITS_JSON="logs/api/logits_submission_fast_20251123_024816.json"
    echo "✓ Found logits JSON: $LOGITS_JSON"
elif [ -n "$(ls logs/api/logits_submission_fast_*.json 2>/dev/null)" ]; then
    LOGITS_JSON=$(ls -t logs/api/logits_submission_fast_*.json | head -1)
    echo "✓ Found logits JSON: $LOGITS_JSON"
else
    echo "⚠ Warning: No logits JSON found. Will auto-detect or generate."
    echo "  If auto-detection fails, run:"
    echo "    python analyze.py output/submission_fast.npz --mode api"
fi

# Run Lock & Ram strategy
# Parameters:
# - epsilon: 8.0 (fixed, no binary search)
# - kappa: 50.0 (huge margin requirement)
# - pgd_steps: 150 (increased for better convergence to kappa=50.0)
# - restarts: 5 (avoid local minima)
# - Input Diversity: ENABLED (default)

python -u lock_and_ram.py \
    --previous-submission "$PREVIOUS_SUB" \
    --logits-json "$LOGITS_JSON" \
    --dataset ../natural_images.pt \
    --output-dir ./output \
    --log-dir ./logs \
    --output-name submission_ram_test.npz \
    --epsilon 8.0 \
    --kappa 30.0 \
    --pgd-steps 150 \
    --restarts 5 \
    --alpha-factor 2.5 \
    --momentum 0.9 \
    --device cuda

exit_code=$?

echo ""
echo "=================================================================="
echo "Job completed: $(date)"
echo "Exit code: $exit_code"
echo "=================================================================="

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ Lock & Ram strategy completed successfully!"
    echo ""
    echo "=================================================================="
    echo "SUBMISSION INSTRUCTIONS"
    echo "=================================================================="
    echo ""
    echo "1. Review results locally (fast):"
    echo "   python analyze.py output/submission_ram_test.npz --mode local"
    echo ""
    echo "2. Test transfer with API (wait 15 min after last query if needed):"
    echo "   python analyze.py output/submission_ram_test.npz --mode api"
    echo ""
    echo "3. Submit to leaderboard (wait 5 min after last submission if needed):"
    echo "   python submit.py output/submission_ram_test.npz --action submit"
    echo ""
    echo "4. Check leaderboard:"
    echo "   http://34.122.51.94:80/leaderboard_page"
    echo ""
    echo "=================================================================="
    echo "NOTE: API rate limits:"
    echo "  - Logits query: 1 every 15 minutes"
    echo "  - Submission: 1 every 5 minutes"
    echo "=================================================================="
else
    echo ""
    echo "❌ Lock & Ram strategy failed with exit code $exit_code"
    echo "Check logs/slurm_${SLURM_JOB_ID}.err for details"
fi

exit $exit_code

