# Task 1: Adversarial Examples - Context Summary

## Objective
Create 100 adversarial examples from `natural_images.pt` that:
1.  Look identical to originals (minimize L2 distance).
2.  Are misclassified by the remote "Black Box" model.
3.  **Score:** Average L2 distance (Lower is better). Failed attacks (correct classification) = 1.0 penalty.

## Infrastructure & Environment
*   **Cluster:** JURECA (SLURM).
*   **Nodes:** 
    *   Login nodes (coding, lightweight scripts).
    *   Compute nodes (`dc-gpu`, `dc-gpu-devel`) for heavy attacks (PGD).
*   **Env:** Requires `module load GCC CUDA PyTorch torchvision`.
*   **Rate Limits:**
    *   Logits Query (`GET_LOGITS`): 15 min cooldown.
    *   Submission (`SUBMIT`): 5 min cooldown.

## Workflow
1.  **Attack Generation (`sandbox.py` / `run_pgd_attack.py`):**
    *   Load `natural_images.pt`.
    *   Generate adversarial images (e.g., PGD on local Surrogate Model like ResNet18).
    *   **Constraint:** Output `.npz` must contain `'images'` (float32, [0,1], shape 100x3x28x28) AND `'image_ids'`.
    *   Save as `submission.npz`.
2.  **Submission (`task_template.py`):**
    *   Set `FILE_PATH = "submission.npz"`.
    *   Set `SUBMIT = True`.
    *   Run to upload.

## Key Files
*   `natural_images.pt`: Dataset (Images, IDs, True Labels).
*   `task_template.py`: Official submission script (do not use for generation).
*   `sandbox.py`: Current generation script (creates `simple_attack.npz`).
*   `run_attack.sh`: SLURM script to request GPU.
*   `vis_output/`: Visualization of generated images.

## Learned Constraints
*   **Black Box:** We cannot know the model. Use Transfer Attacks (attack local ResNet/VGG -> submit).
*   **API:** Separate "Logits" (Test) and "Submit" (Score) endpoints.
*   **Env:** Must load modules before running python scripts.

