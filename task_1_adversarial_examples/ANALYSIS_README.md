# Task 1: Analysis & Visualization

## 1. Analyze your Submission
Instead of `task_template.py`, use this new script. It queries the API, gets the logits, saves them to a log file, and calculates your Score.

```bash
module load GCC CUDA PyTorch torchvision
python analyze_submission.py
```

It will print:
*   Success Rate (Did we fool them?)
*   Average L2 Distance (How noisy?)
*   **Estimated Score** (The number on the leaderboard)

## 2. Visualize Results
Open `results_viewer.ipynb` in Jupyter/Cursor.
*   It automatically loads the latest log from `logs/`.
*   It shows images where you can see the noise.
*   It shows the model's predictions vs true labels.

## 3. Tuning Epsilon
Edit `EPSILON_CONFIG.py` to change the attack strength.
*   **Current:** 0.5
*   **Workflow:**
    1.  Change `EPSILON` in `EPSILON_CONFIG.py`.
    2.  Run `python run_pgd.py` (Generates images).
    3.  Run `python analyze_submission.py` (Checks score).
    4.  If Success Rate is 100%, **LOWER** Epsilon.
    5.  If Success Rate < 100%, **RAISE** Epsilon.
EDIT: Epsilon is changed directly in run_pgd.py

