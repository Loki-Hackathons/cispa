# JURECA Compute Resources Summary

Based on `Docs/presentation_slides.txt` and `Docs/gpu_acces.txt`.

## Available Hardware

### Compute Nodes (40 Total)
*   **Resources per Node:**
    *   **GPU:** 4× NVIDIA A100 (40 GB Memory each) - *Very Powerful*
    *   **CPU:** 2× AMD EPYC 7742 (128 cores total)
    *   **RAM:** 512 GB
*   **Usage:** Heavy computations (Training, Attacking).
*   **Access:** Only via SLURM (`sbatch` or `salloc`).

### Login Nodes (12 Total)
*   **Resources:**
    *   **GPU:** 2× NVIDIA Quadro RTX8000
    *   **CPU:** 2× AMD EPYC 7742
*   **Usage:** Coding, file management, light testing, submission.
*   **Access:** Direct SSH terminal. **DO NOT run heavy jobs here.**

## Requesting Resources (SLURM)

### Time Limits
*   **Debug/Devel:** Short jobs (usually < 2 hours). Good for testing.
*   **Production:** Longer jobs (typically up to 24 hours, check specific partition limits with `sinfo`).

### Standard Command (`run_attack.sh`)
The partition `dc-gpu-devel` is likely for development/testing.

```bash
#SBATCH --partition=dc-gpu-devel  # Partition Name
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --gres=gpu:1              # Number of GPUs (1 to 4)
#SBATCH --time=02:00:00           # Max duration (HH:MM:SS)
```

### Partitions (Queues)
*   `dc-gpu`: Standard GPU partition.
*   `dc-gpu-devel`: Development partition (shorter time limits, faster start).

## How to Run
1.  **Modify Script:** Edit `#SBATCH` lines in `.sh` file.
2.  **Submit:** `sbatch run_attack.sh`
3.  **Check:** `squeue -u $USER`

