# Task 1 Setup Guide

## Step 1: Environment Setup

### On Login Node (jrlogin05 or similar)

```bash
# Load required modules
module load GCC
module load CUDA
module load PyTorch
module load torchvision

# Activate project
jutil env activate -p training2557

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Configure uv (add to ~/.bashrc)
cat << EOF >> ~/.bashrc
export UV_PYTHON_INSTALL_DIR=$PROJECT/user_dirs/$USER/uv/python
export UV_CACHE_DIR=$PROJECT/user_dirs/$USER/uv/cache
export UV_TOOL_DIR=$PROJECT/user_dirs/$USER/uv/tools
EOF

# Restart shell
exec bash

# Navigate to your project
cd /p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples

# Create Python environment
uv venv -p 3.12
source .venv/bin/activate

# Install dependencies
uv pip install torch torchvision numpy requests
```

## Step 2: Test Dataset Loading

```bash
# On login node (no GPU needed for this)
python task_template.py
```

This should print dataset information. Verify `natural_images.pt` loads correctly.

## Step 3: Create SLURM Script for GPU Jobs

Create `run_attack.sh`:

```bash
#!/bin/bash
#SBATCH --account=training2557
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu-devel
#SBATCH --output=output/%j.out
#SBATCH --time=02:00:00

# Activate environment
source /p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples/.venv/bin/activate

# Run your attack script
python task_template.py
```

Make it executable:
```bash
chmod +x run_attack.sh
```

## Step 4: Submit GPU Job

```bash
sbatch run_attack.sh
```

Check job status:
```bash
squeue -u $USER
```

View output:
```bash
tail -f output/<job_id>.out
```

## Step 5: Development Workflow

1. **Develop locally** (login node): Write and test code logic
2. **Test API queries** (login node): Query API to understand classifier behavior
3. **Run attacks** (GPU node): Use `sbatch run_attack.sh` for actual adversarial generation
4. **Submit results**: Once you have adversarial examples, submit via API

## Important Notes

- **API Rate Limits**: 
  - Query logits: Once every 15 minutes
  - Submit results: Once every 5 minutes
  
- **File Format**: 
  - Submission must be `.npz` with `'images'` key
  - Shape: `(100, 3, 28, 28)`
  - Dtype: `float32`
  - Values: `[0, 1]` range (normalized)
  - No NaN or Inf values

- **Dataset Location**: 
  - Local: `task_1_adversarial_examples/natural_images.pt`
  - Shared: `/p/project1/training2557/common/adversarial-examples`

