# SLURM Resource Allocation Guide - JURECA

## Understanding SLURM

**SLURM** (Simple Linux Utility for Resource Management) is the job scheduler on JURECA. You **cannot** run heavy GPU jobs directly on login nodes - you must request resources first.

---

## Two Ways to Use SLURM

### Method 1: Interactive (`salloc` + `srun`) - For Testing

**Step 1: Allocate Resources**
```bash
salloc -p dc-gpu -t 20 -N 1 -A training2557 --gres=gpu:1
```
- `-p dc-gpu`: Partition (queue) - `dc-gpu` has A100 GPUs
- `-t 20`: Time limit in minutes (20 min)
- `-N 1`: 1 compute node
- `-A training2557`: Your account/project
- `--gres=gpu:1`: Request 1 GPU

**Step 2: Use the Allocated Node**
```bash
srun --pty bash -i
```
- This gives you an interactive shell **on the compute node**
- Now you can run Python scripts that use GPU

**Step 3: Run Your Script**
```bash
module load GCC CUDA PyTorch torchvision
cd /p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples
python run_pgd_ensemble_simple.py
```

**Step 4: Exit**
```bash
exit  # Exit the compute node shell
exit  # Release the allocation
```

---

### Method 2: Batch Job (`sbatch`) - For Production

**Step 1: Create Script with #SBATCH Directives**
```bash
#!/bin/bash
#SBATCH --account=training2557
#SBATCH --partition=dc-gpu          # Use dc-gpu for A100 GPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00              # 4 hours
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

# Your commands here
module load GCC CUDA PyTorch torchvision
python run_pgd_ensemble_simple.py
```

**Step 2: Submit Job**
```bash
sbatch run_attack.sh
```

**Step 3: Monitor**
```bash
squeue -u $USER              # See your jobs
tail -f output/<job_id>.out  # Watch output
```

---

## Partitions (Queues)

- **`dc-gpu`**: Production partition with A100 GPUs (4 per node)
  - Longer time limits
  - May have wait time
  
- **`dc-gpu-devel`**: Development partition
  - Faster start (less wait)
  - Shorter time limits (usually 2 hours max)
  - Good for testing

---

## GPU Types

- **Login Nodes**: Quadro RTX 8000 (weaker, for light testing)
- **Compute Nodes**: NVIDIA A100 (40GB) - **This is what you want!**
  - 4× A100 per compute node
  - Much faster than RTX 8000

---

## Key Commands

```bash
# Check available partitions
sinfo

# Check your jobs
squeue -u $USER

# Cancel a job
scancel <job_id>

# Check job details
scontrol show job <job_id>

# Check GPU availability
sinfo -p dc-gpu -o "%N %G %T"
```

---

## Example: Requesting A100 GPU

### Interactive (for testing):
```bash
salloc -p dc-gpu -t 60 -N 1 -A training2557 --gres=gpu:1
srun --pty bash -i
# Now you're on a compute node with A100 GPU
nvidia-smi  # Verify GPU
python your_script.py
exit
exit
```

### Batch (for production):
```bash
sbatch run_attack.sh  # Script already has #SBATCH directives
```

---

## Important Notes

1. **Login Nodes**: Only for coding, file management, light testing
2. **Compute Nodes**: For heavy GPU work (accessed via SLURM)
3. **Time Limits**: Jobs are killed when time limit expires
4. **Resource Limits**: Don't request more than you need (slower scheduling)
5. **Output Files**: Check `output/` directory for job logs

---

## Troubleshooting

### Job Stuck in Queue with "(Nodes required for job are DOWN, DRAINED or reserved...)"

**Problem:** The `dc-gpu` partition may have many nodes down or reserved.

**Solution:** Switch to `dc-gpu-devel` partition:
```bash
# In your .sh script, change:
#SBATCH --partition=dc-gpu-devel  # Instead of dc-gpu
#SBATCH --time=02:00:00           # Max 2 hours for devel partition
```

**Check partition status:**
```bash
sinfo -p dc-gpu -o "%P %a %l %D %T %N"
sinfo -p dc-gpu-devel -o "%P %a %l %D %T %N"
```

### "QOSMaxWallDurationPerJobLimit" Error

**Problem:** Time limit too long for the partition.

**Solution:** Reduce `--time` parameter:
- `dc-gpu-devel`: Max 2 hours (`--time=02:00:00`)
- `dc-gpu`: Longer limits available (check with `sinfo`)

### No Output Appearing in Log File

**Problem:** Python buffers stdout, so output doesn't appear immediately.

**Solution:** Use unbuffered Python:
```bash
# In your .sh script:
python -u your_script.py  # -u flag forces unbuffered output
```

Or add `flush=True` to print statements in Python:
```python
print("Message", flush=True)
```

### Job Status Codes

- **PD** (Pending): Waiting in queue
- **CF** (Configuring): Starting up
- **R** (Running): Active
- **CG** (Completing): Finishing
- **CD** (Completed): Finished successfully
- **F** (Failed): Error occurred

---

## Real-World Example: Running Ensemble PGD Attack

### Step 1: Check Available Resources
```bash
sinfo -p dc-gpu-devel
```

### Step 2: Submit Job
```bash
cd /p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples
sbatch run_attack.sh
```

### Step 3: Monitor Progress
```bash
# Check job status
squeue -u $USER

# Watch output in real-time (once job starts)
tail -f output/<job_id>.out

# Check for errors
tail -f output/<job_id>.err
```

### Step 4: Verify Results
```bash
# Check if submission file was created
ls -lh submission_pgd.npz

# Analyze results locally
python analyze_submission.py submission_pgd.npz --mode local

# Or query API for actual logits
python analyze_submission.py submission_pgd.npz --mode api
```

---

## Key Learnings

1. **Always use `dc-gpu-devel` for development/testing** - Faster start, sufficient for most tasks
2. **Use `python -u` for real-time logging** - Critical for monitoring long-running jobs
3. **Check partition status before submitting** - Avoid queues with drained nodes
4. **Monitor output files** - Use `tail -f` to watch progress
5. **Time limits matter** - Devel partition has stricter limits (2h max)

